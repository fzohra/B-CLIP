import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class TextConditionedImageLoss(nn.Module):
    def __init__(
        self,
        cls_alpha: float = 1.0,
        tcil_alpha: float = 1.0,
        mps_alpha: float = 1.0,
        tcil_neg_alpha: float = 10.0,
        tci_diversity_alpha: float = 0.5,
        use_cls_loss:bool = False,
        use_tcil_loss:bool = False,
        use_mps_loss:bool = False,
        use_tci_diversity_loss: bool = False,
        use_diversity_hinge: bool = False,
        tci_div_separate: bool = False,
        use_negative_text_conditioning: bool = False,
        tcil_loss_mode: str = "k_positives_ce",
        use_softmax_for_multi_positives: bool = False,
        num_caption_tokens: int = 1,
        num_eos_tokens: int = 0,
        num_concept_tokens: int = 0,
        alpha: float = 1.0,  # Weight for diagonal (self-matching) in alpha-beta interpolation
        beta: float = 0.0,   # Weight for off-diagonal within-image in alpha-beta interpolation
    ):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

        self.cls_alpha = cls_alpha
        self.tcil_alpha = tcil_alpha
        self.mps_alpha = mps_alpha
        self.tci_diversity_alpha = tci_diversity_alpha
        self.tcil_neg_alpha = tcil_neg_alpha
        
        self.use_cls_loss = use_cls_loss
        self.use_tcil_loss = use_tcil_loss
        self.use_mps_loss = use_mps_loss
        self.use_negative_text_conditioning = use_negative_text_conditioning
        self.tcil_loss_mode = tcil_loss_mode
        self.use_softmax_for_multi_positives = use_softmax_for_multi_positives
        
        self.alpha = alpha
        self.beta = beta
        
        self.use_tci_diversity_loss = use_tci_diversity_loss
        self.num_caption_tokens = num_caption_tokens
        self.num_eos_tokens = num_eos_tokens
        self.num_concept_tokens = num_concept_tokens
        self.tci_div_separate = tci_div_separate
        self.use_diversity_hinge = use_diversity_hinge


    def get_global_loss(self, eos_embed, cls_embed, logit_scale):
        local_batch_size = cls_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=cls_embed.device
            )
            self.last_local_batch_size = local_batch_size

        eos_embed_n = F.normalize(eos_embed, dim=-1, p=2)
        cls_embed_n = F.normalize(cls_embed, dim=-1, p=2)

        image_all, text_all = utils.all_gather_batch([cls_embed_n, eos_embed_n])

        logits_per_image = logit_scale * (cls_embed_n @ text_all.t())   # [B, B_total]
        logits_per_text  = logit_scale * (eos_embed_n  @ image_all.t()) # [B, B_total]
        
        cls_eos_loss = 0.5 * (
            F.cross_entropy(logits_per_image, self.labels) +
            F.cross_entropy(logits_per_text,  self.labels)
        )

        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            cls_eos_acc = 100 * correct / local_batch_size

        return cls_eos_loss, cls_eos_acc

    def compute_diversity(self, tci, mask):
        B, K, D = tci.shape
        
        tci_norm = F.normalize(tci, dim=-1)
        sim = torch.bmm(tci_norm, tci_norm.transpose(1, 2)) # [B, K, K]
        
        eye_mask = torch.eye(K, device=tci.device, dtype=torch.bool).unsqueeze(0)
        valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        off_diag_mask = ~eye_mask & valid_mask
        
        off_diag_sims = sim[off_diag_mask]
        
        diversity_metric = off_diag_sims.mean()
        
        return diversity_metric

    def get_mps_loss(self, text_embed, text_mask, cls_embed, logit_scale): # k_positives mode
        """
        Compute symmetric cross entropy loss for text-conditioned image embeddings.
        Uses all_gather to increase batch size across distributed processes.
        
        Args:
            text_embed: [B, K, D] - text embeddings (local batch)
            text_mask: [B, K] - mask for valid text embeddings (local batch)
            img_embed: [B, D] - text-conditioned image embeddings (local batch)
            logit_scale: scalar - temperature scaling
        
        Returns:
            tcil_loss: scalar tensor
            tcil_acc: scalar tensor
        """
        B, K, D = text_embed.shape
        local_batch_size = B
        rank = utils.get_rank()
        
        # Normalize embeddings
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        cls_embed = F.normalize(cls_embed, dim=-1, p=2)
        
        # All-gather embeddings across all processes to get global batch
        # text_embed_all: [B_total, K, D], img_embed_all: [B_total, D], text_mask_all: [B_total, K]
        # Ensure tensors are contiguous for NCCL
        gathered = utils.all_gather_batch([text_embed.contiguous(), cls_embed.contiguous(), text_mask.contiguous()])
        text_embed_all = gathered[0]  # [B_total, K, D]
        img_embed_all = gathered[1]   # [B_total, D]
        text_mask_all = gathered[2]    # [B_total, K]
        
        B_total = text_embed_all.shape[0]
        
        # Reshape to flat for cross-batch comparisons
        # Use reshape instead of view to handle non-contiguous tensors
        text_embed_flat_all = text_embed_all.reshape(B_total * K, D)  # [B_total*K, D]
        text_mask_flat_all = text_mask_all.reshape(B_total * K)  # [B_total*K]
        
        # Compute similarity matrix between LOCAL embeddings and ALL gathered embeddings
        # This gives us cross-batch negatives
        logits_per_image = logit_scale * (cls_embed @ text_embed_flat_all.t())  # [B, B_total*K]
        logits_per_text = logit_scale * (text_embed.reshape(B * K, D) @ img_embed_all.t())  # [B*K, B_total]
        
        # K_positive approach with softmax (with distributed gathering):
        # For LOCAL image i (rank*B + i globally), the K positive texts are at GLOBAL indices 
        # [(rank*B + i)*K : (rank*B + i)*K + K] in the flattened all-gathered text tensor
        # For LOCAL text at position b*K+k, the positive image is at GLOBAL index rank*B + b
        
        # Create positive mask for image-to-text: [B_local, B_total*K]
        # Local image i corresponds to global batch index (rank*B + i)
        local_img_indices = torch.arange(B, device=text_embed.device) + rank * B  # [B] - global indices
        global_text_batch_indices = torch.arange(B_total * K, device=text_embed.device) // K  # [B_total*K] - which image each text belongs to
        positive_mask_img = (local_img_indices.unsqueeze(1) == global_text_batch_indices.unsqueeze(0)).float()  # [B, B_total*K]
        
        # Apply text mask to positive_mask_img
        positive_mask_img = positive_mask_img * text_mask_flat_all.unsqueeze(0).float()  # [B, B_total*K]
        
        # Create positive mask for text-to-image: [B_local*K, B_total]
        # Local text at position b*K+k corresponds to global batch index (rank*B + b)
        local_text_batch_indices = torch.arange(B * K, device=text_embed.device) // K + rank * B  # [B*K] - global batch indices
        global_img_indices = torch.arange(B_total, device=text_embed.device)  # [B_total]
        positive_mask_text = (local_text_batch_indices.unsqueeze(1) == global_img_indices.unsqueeze(0)).float()  # [B*K, B_total]
        
        # Apply text mask to positive_mask_text (only valid text embeddings contribute)
        # Need to use local text mask for local texts
        text_mask_flat_local = text_mask.reshape(B * K)  # [B*K]
        positive_mask_text = positive_mask_text * text_mask_flat_local.unsqueeze(1).float()  # [B*K, B_total]
        
        # Softmax with multiple positives
        # Image-to-text loss (local images vs all texts)
        log_probs_image = F.log_softmax(logits_per_image, dim=-1)  # [B, B_total*K]
        
        # Sum log probabilities of positive pairs, normalized by number of positives per image
        num_positives_per_image = positive_mask_img.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1]
        loss_image = -(log_probs_image * positive_mask_img).sum(dim=-1) / num_positives_per_image.squeeze(-1)  # [B]
        loss_image = loss_image.mean()  # scalar
        
        # Text-to-image loss (local texts vs all images)
        log_probs_text = F.log_softmax(logits_per_text, dim=-1)  # [B*K, B_total]
        
        # Sum log probabilities of positive pairs (each text has 1 positive image)
        loss_text = -(log_probs_text * positive_mask_text).sum(dim=-1)  # [B*K]
        
        # Average only over valid text embeddings (use local text mask)
        valid_text_mask = text_mask_flat_local  # [B*K]
        loss_text = loss_text[valid_text_mask].mean() if valid_text_mask.sum() > 0 else torch.tensor(0.0, device=text_embed.device)
        
        # Total loss (symmetric)
        tcil_loss = 0.5 * (loss_image + loss_text)
        
        # Compute accuracy (only for local batch)
        with torch.no_grad():
            # For image-to-text: check if highest probability is assigned to one of the K positives
            pred_image = torch.argmax(logits_per_image, dim=-1)  # [B]
            correct_image = positive_mask_img[torch.arange(B, device=text_embed.device), pred_image]  # [B]
            correct_image = correct_image.sum()
            
            # For text-to-image: check if highest probability is assigned to the positive image
            pred_text = torch.argmax(logits_per_text, dim=-1)  # [B*K]
            correct_text = positive_mask_text[torch.arange(B * K, device=text_embed.device), pred_text]  # [B*K]
            correct_text = correct_text[valid_text_mask].sum()
            
            # Total accuracy (based on local batch)
            total_valid = B + valid_text_mask.sum()
            if total_valid > 0:
                tcil_acc = 100 * (correct_image + correct_text) / total_valid
            else:
                tcil_acc = torch.tensor(0.0, device=text_embed.device)
        
        return tcil_loss, tcil_acc

    def get_text_conditioned_image_loss(
        self,
        text_embed,
        text_mask,
        positive_img_embed,
        logit_scale_tc,
        negative_text_embed=None,
        negative_img_embed=None,
    ):
        """
        Compute symmetric cross entropy loss for text-conditioned image embeddings.
        
        Args:
            text_embed: [B, K, D] - text embeddings
            text_mask: [B, K] - mask for valid text embeddings
            positive_img_embed: [B, K, D] - text-conditioned image embeddings
            logit_scale_tc: scalar - temperature scaling
        
        Returns:
            tcil_loss: scalar tensor
            tcil_acc: scalar tensor
        
        Optional Args (used by k_positives_neg_ce mode):
            negative_text_embed: [B, N, D] - explicit negative text embeddings
            negative_img_embed: [B, N, D] - explicit negative image embeddings
        """
        B, K, D = text_embed.shape
        
        # Normalize embeddings
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        positive_img_embed = F.normalize(positive_img_embed, dim=-1, p=2)
        
        # Reshape to [B*K, D] for cross-batch comparisons
        # Use reshape instead of view to handle non-contiguous tensors
        text_embed_flat = text_embed.reshape(B * K, D)  # [B*K, D]
        img_embed_flat = positive_img_embed.reshape(B * K, D)  # [B*K, D]
        text_mask_flat = text_mask.reshape(B * K)  # [B*K]
        
        # Compute similarity matrix between all image and text embeddings
        logits_per_image = logit_scale_tc * (img_embed_flat @ text_embed_flat.t())  # [B*K, B*K]
        logits_per_text = logit_scale_tc * (text_embed_flat @ img_embed_flat.t())  # [B*K, B*K]
        
        if self.tcil_loss_mode == "k_positives_bce":
            # All within-image positives use hard targets (1.0), but we scale the loss
            # contribution of off-diagonal positives by beta instead of softening targets.
            off_diag_weight = torch.clamp(torch.tensor(self.beta, device=text_embed.device), min=0.0)

            # Block-diagonal positive mask (1s for within-image pairs, 0 elsewhere)
            eye_batch = torch.eye(B, device=text_embed.device)
            ones_block = torch.ones(K, K, device=text_embed.device)
            target_weights = torch.kron(eye_batch, ones_block)

            # Ensure diagonal positives remain at 1.0
            target_weights.fill_diagonal_(1.0)

            # Apply text mask to drop padded tokens
            text_mask_2d = text_mask_flat.unsqueeze(0) & text_mask_flat.unsqueeze(1)
            target_weights = target_weights * text_mask_2d.float()

            # Build element-wise loss weights: scale within-image off-diagonals by beta
            loss_weights = torch.ones_like(target_weights)
            same_image_mask = torch.kron(
                eye_batch.bool(), torch.ones(K, K, device=text_embed.device, dtype=torch.bool)
            )
            diagonal_mask = torch.eye(B * K, device=text_embed.device, dtype=torch.bool)
            off_diag_positive_mask = same_image_mask & (~diagonal_mask)
            loss_weights = loss_weights.masked_fill(off_diag_positive_mask, off_diag_weight)
            loss_weights.fill_diagonal_(1.0)
            loss_weights = loss_weights * text_mask_2d.float()

            NEG = -1e4 if logits_per_image.dtype == torch.float16 else -1e9
            logits_per_image = logits_per_image.masked_fill(~text_mask_flat.unsqueeze(0), NEG)
            logits_per_text = logits_per_text.masked_fill(~text_mask_flat.unsqueeze(0), NEG)

            # BCE loss with weighted contributions
            loss_image = F.binary_cross_entropy_with_logits(
                logits_per_image, target_weights, weight=loss_weights
            )
            loss_text = F.binary_cross_entropy_with_logits(
                logits_per_text, target_weights, weight=loss_weights
            )

            tcil_loss = 0.5 * (loss_image + loss_text)

            with torch.no_grad():
                pred_image = (torch.sigmoid(logits_per_image) > 0.5).float()
                pred_text = (torch.sigmoid(logits_per_text) > 0.5).float()

                targets = (target_weights > 0).float()

                correct_image = (pred_image[text_mask_2d] == targets[text_mask_2d]).sum()
                correct_text = (pred_text[text_mask_2d] == targets[text_mask_2d]).sum()
                total = text_mask_2d.sum()

                if total > 0:
                    tcil_acc = 100 * (correct_image + correct_text) / (2 * total)
                else:
                    tcil_acc = torch.tensor(0.0, device=text_embed.device)

        elif self.tcil_loss_mode == "k_positives_neg_ce":            
            negative_text_embed = F.normalize(negative_text_embed, dim=-1, p=2)
            negative_img_embed = F.normalize(negative_img_embed, dim=-1, p=2)
            
            B_neg = negative_text_embed.shape[1]
            if negative_img_embed.shape[1] != B_neg:
                raise ValueError(
                    "negative_text_embed and negative_img_embed must have the same number of negatives"
                )
            
            if B_neg > 0:
                neg_text_flat = negative_text_embed.reshape(B * B_neg, D)  # [B*N, D]
                neg_img_flat = negative_img_embed.reshape(B * B_neg, D)    # [B*N, D]
                
                logits_neg_image = logit_scale_tc * (img_embed_flat @ neg_text_flat.t())  # [B*K, B*N]
                logits_neg_text = logit_scale_tc * (text_embed_flat @ neg_img_flat.t())   # [B*K, B*N]
                
                logits_per_image_ext = torch.cat([logits_per_image, logits_neg_image], dim=1)  # [B*K, B*(K+N)]
                logits_per_text_ext = torch.cat([logits_per_text, logits_neg_text], dim=1)      # [B*K, B*(K+N)]
            else:
                logits_per_image_ext = logits_per_image
                logits_per_text_ext = logits_per_text
            
            batch_indices = torch.arange(B * K, device=text_embed.device) // K  # [B*K]
            same_image_mask = (batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0)).float()  # [B*K, B*K]
            
            diagonal_indices = torch.arange(B * K, device=text_embed.device)
            diagonal_mask = torch.zeros(B * K, B * K, device=text_embed.device)
            diagonal_mask[diagonal_indices, diagonal_indices] = 1.0
            
            off_diagonal_within_image_mask = same_image_mask * (1.0 - diagonal_mask)
            
            target_weights = self.alpha * diagonal_mask + self.beta * off_diagonal_within_image_mask
            text_mask_2d = text_mask_flat.unsqueeze(0) & text_mask_flat.unsqueeze(1)  # [B*K, B*K]
            target_weights = target_weights * text_mask_2d.float()
            
            if B_neg > 0:
                zero_block = torch.zeros(
                    B * K,
                    B * B_neg,
                    device=text_embed.device,
                    dtype=target_weights.dtype,
                )
                target_weights_ext = torch.cat([target_weights, zero_block], dim=1)  # [B*K, B*(K+N)]
            else:
                target_weights_ext = target_weights
            
            row_sums = target_weights_ext.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            target_distribution = target_weights_ext / row_sums
            
            log_probs_image = F.log_softmax(logits_per_image_ext, dim=-1)  # [B*K, B*(K+N)]
            log_probs_text = F.log_softmax(logits_per_text_ext, dim=-1)    # [B*K, B*(K+N)]
            
            loss_image = -(target_distribution * log_probs_image).sum(dim=-1)  # [B*K]
            loss_text = -(target_distribution * log_probs_text).sum(dim=-1)    # [B*K]
            
            valid_rows = text_mask_flat  # [B*K]
            if valid_rows.sum() > 0:
                loss_image = loss_image[valid_rows].mean()
                loss_text = loss_text[valid_rows].mean()
                tcil_loss = 0.5 * (loss_image + loss_text)
            else:
                zero = torch.tensor(0.0, device=text_embed.device, dtype=loss_image.dtype)
                loss_image = zero
                loss_text = zero
                tcil_loss = zero
            
            with torch.no_grad():
                pred_image = torch.argmax(logits_per_image_ext, dim=-1)  # [B*K]
                pred_text = torch.argmax(logits_per_text_ext, dim=-1)    # [B*K]
                
                positive_mask_ext = target_weights_ext > 0
                
                correct_image = positive_mask_ext[
                    torch.arange(B * K, device=text_embed.device), pred_image
                ]
                correct_text = positive_mask_ext[
                    torch.arange(B * K, device=text_embed.device), pred_text
                ]
                
                correct_image = correct_image[valid_rows].sum()
                correct_text = correct_text[valid_rows].sum()
                total = valid_rows.sum()
                
                if total > 0:
                    tcil_acc = 100 * (correct_image + correct_text) / (2 * total)
                else:
                    tcil_acc = torch.tensor(0.0, device=text_embed.device)
        elif self.tcil_loss_mode == "k_positives_ce":
            # Create batch indices: [0,0,0,1,1,1,2,2,2,...] for B=3, K=3
            batch_indices = torch.arange(B * K, device=text_embed.device) // K  # [B*K]
            
            # Create block diagonal mask (1.0 where batch indices match, 0.0 otherwise)
            same_image_mask = (batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0)).float()  # [B*K, B*K]
            
            # Create diagonal mask
            diagonal_indices = torch.arange(B * K, device=text_embed.device)
            diagonal_mask = torch.zeros(B * K, B * K, device=text_embed.device)
            diagonal_mask[diagonal_indices, diagonal_indices] = 1.0
            
            # Create off-diagonal within-image mask
            off_diagonal_within_image_mask = same_image_mask * (1.0 - diagonal_mask)
            
            # Create weighted target: alpha * diagonal + beta * off_diagonal_within_image
            target_weights = self.alpha * diagonal_mask + self.beta * off_diagonal_within_image_mask
            # Apply text mask: only valid text embeddings are considered
            text_mask_2d = text_mask_flat.unsqueeze(0) & text_mask_flat.unsqueeze(1)  # [B*K, B*K]
            target_weights = target_weights * text_mask_2d.float()
            
            # Normalize each row to sum to 1 (valid probability distribution for cross-entropy)
            row_sums = target_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B*K, 1]
            target_distribution = target_weights / row_sums  # [B*K, B*K]
            # Compute log softmax for both directions
            log_probs_image = F.log_softmax(logits_per_image, dim=-1)  # [B*K, B*K]
            log_probs_text = F.log_softmax(logits_per_text, dim=-1)    # [B*K, B*K]
            
            # Cross-entropy loss: -sum(target * log_prob)
            loss_image = -(target_distribution * log_probs_image).sum(dim=-1)  # [B*K]
            loss_text = -(target_distribution * log_probs_text).sum(dim=-1)    # [B*K]
            
            # Average over valid pairs
            valid_rows = text_mask_flat  # [B*K]
            loss_image = loss_image[valid_rows].mean()
            loss_text = loss_text[valid_rows].mean()
            
            tcil_loss = 0.5 * (loss_image + loss_text)
            
            # Compute accuracy: check if highest probability is assigned to a positive (weight > 0)
            with torch.no_grad():
                pred_image = torch.argmax(logits_per_image, dim=-1)  # [B*K]
                pred_text = torch.argmax(logits_per_text, dim=-1)    # [B*K]
                
                # Check if prediction has non-zero weight in target
                is_positive_image = (target_weights[torch.arange(B * K, device=text_embed.device), pred_image] > 0)
                is_positive_text = (target_weights[torch.arange(B * K, device=text_embed.device), pred_text] > 0)
                
                correct_image = is_positive_image[valid_rows].sum()
                correct_text = is_positive_text[valid_rows].sum()
                total = valid_rows.sum()
                
                if total > 0:
                    tcil_acc = 100 * (correct_image + correct_text) / (2 * total)
                else:
                    tcil_acc = torch.tensor(0.0, device=text_embed.device)

        elif self.tcil_loss_mode == "k_positives_ce_reduced":
            # Same as k_positives_ce but masks concept COLUMNS in cross-batch negatives
            # This allows: sent→sent and concept→sent as negatives
            # But masks: sent→concept and concept→concept            
            # Create batch indices: [0,0,0,1,1,1,2,2,2,...] for B=3, K=3
            batch_indices = torch.arange(B * K, device=text_embed.device) // K  # [B*K]
            
            # Create block diagonal mask (1.0 where batch indices match, 0.0 otherwise)
            same_image_mask = (batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0)).float()  # [B*K, B*K]
            
            # Create diagonal mask
            diagonal_indices = torch.arange(B * K, device=text_embed.device)
            diagonal_mask = torch.zeros(B * K, B * K, device=text_embed.device)
            diagonal_mask[diagonal_indices, diagonal_indices] = 1.0
            
            # Create off-diagonal within-image mask
            off_diagonal_within_image_mask = same_image_mask * (1.0 - diagonal_mask)
            
            # Create weighted target: alpha * diagonal + beta * off_diagonal_within_image
            target_weights = self.alpha * diagonal_mask + self.beta * off_diagonal_within_image_mask
            
            # Apply text mask: zero out invalid/padded positions in targets
            text_mask_2d = text_mask_flat.unsqueeze(0) & text_mask_flat.unsqueeze(1)  # [B*K, B*K]
            target_weights = target_weights * text_mask_2d.float()
            
            # Mask invalid positions in logits (consistent with 1_positive mode)
            NEG = -1e4 if logits_per_image.dtype == torch.float16 else -1e9
            
            # First, mask out invalid/padded text positions
            logits_per_image_masked = logits_per_image.masked_fill(~text_mask_flat.unsqueeze(0), NEG)
            logits_per_text_masked = logits_per_text.masked_fill(~text_mask_flat.unsqueeze(0), NEG)
            
            # Then, mask cross-batch concept columns
            cross_batch_concept_mask = None
            if hasattr(self, 'num_concept_tokens') and self.num_concept_tokens > 0:
                # Identify concept tokens (last num_concept_tokens in each group of K)
                indices = torch.arange(B * K, device=text_embed.device)
                position_in_group = indices % K
                concept_mask = position_in_group >= (K - self.num_concept_tokens)  # [B*K]
                
                # Create mask for concept columns (mask when target/column is a concept)
                # This allows: concept → sent (concepts can use sentences as negatives)
                # But masks: sent → concept and concept → concept
                concept_column_mask = concept_mask.unsqueeze(0)  # [1, B*K] broadcasts to [B*K, B*K]
                
                # Identify cross-batch pairs where the target (column) is a concept
                cross_batch_concept_mask = (~same_image_mask.bool()) & concept_column_mask  # [B*K, B*K]
                
                # Mask cross-batch concepts in logits
                logits_per_image_masked = logits_per_image_masked.masked_fill(cross_batch_concept_mask, NEG)
                logits_per_text_masked = logits_per_text_masked.masked_fill(cross_batch_concept_mask, NEG)
            
            # Normalize each row to sum to 1 (valid probability distribution for cross-entropy)
            row_sums = target_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B*K, 1]
            target_distribution = target_weights / row_sums  # [B*K, B*K]
            
            # Compute log softmax for both directions (using masked logits)
            log_probs_image = F.log_softmax(logits_per_image_masked, dim=-1)  # [B*K, B*K]
            log_probs_text = F.log_softmax(logits_per_text_masked, dim=-1)    # [B*K, B*K]
            
            # Cross-entropy loss: -sum(target * log_prob)
            loss_image = -(target_distribution * log_probs_image).sum(dim=-1)  # [B*K]
            loss_text = -(target_distribution * log_probs_text).sum(dim=-1)    # [B*K]
            
            # Average over valid pairs
            valid_rows = text_mask_flat  # [B*K]
            loss_image = loss_image[valid_rows].mean()
            loss_text = loss_text[valid_rows].mean()
            
            tcil_loss = 0.5 * (loss_image + loss_text)
            
            # Compute accuracy: check if highest probability is assigned to a positive (weight > 0)
            with torch.no_grad():
                pred_image = torch.argmax(logits_per_image_masked, dim=-1)  # [B*K]
                pred_text = torch.argmax(logits_per_text_masked, dim=-1)    # [B*K]
                
                # Check if prediction has non-zero weight in target
                is_positive_image = (target_weights[torch.arange(B * K, device=text_embed.device), pred_image] > 0)
                is_positive_text = (target_weights[torch.arange(B * K, device=text_embed.device), pred_text] > 0)
                
                # Exclude predictions on masked positions (cross-batch concepts)
                if cross_batch_concept_mask is not None:
                    pred_on_masked_image = cross_batch_concept_mask[torch.arange(B * K, device=text_embed.device), pred_image]
                    pred_on_masked_text = cross_batch_concept_mask[torch.arange(B * K, device=text_embed.device), pred_text]
                    is_positive_image = is_positive_image & (~pred_on_masked_image)
                    is_positive_text = is_positive_text & (~pred_on_masked_text)
                
                correct_image = is_positive_image[valid_rows].sum()
                correct_text = is_positive_text[valid_rows].sum()
                total = valid_rows.sum()
                
                if total > 0:
                    tcil_acc = 100 * (correct_image + correct_text) / (2 * total)
                else:
                    tcil_acc = torch.tensor(0.0, device=text_embed.device)


        else:
            raise ValueError(f"Unknown tcil_loss_mode: {self.tcil_loss_mode}")
        
        return tcil_loss, tcil_acc

    def get_tcil_negatives_only(self, logit_scale_tc, negative_text_embed, negative_img_embed):
        B, N, D = negative_text_embed.shape
        
        negative_text_embed = F.normalize(negative_text_embed, dim=-1, p=2)  # [B, N, D]
        negative_img_embed = F.normalize(negative_img_embed, dim=-1, p=2)  # [B, N, D]
        
        # Negative pairs: text[i] <-> image[i] should be dissimilar  
        negative_similarities = torch.sum(negative_text_embed * negative_img_embed, dim=-1)  # [B, N] - dot product
        negative_logits = logit_scale_tc * negative_similarities  # [B, N] 
        negative_targets = torch.zeros_like(negative_logits)  # [B, N] - target = 0 (should be dissimilar)

        negative_mask = torch.ones(B, N, device=negative_text_embed.device, dtype=torch.bool)  # [B, N]

        # Apply mask - only compute loss for valid pairs
        valid_logits = negative_logits[negative_mask]  # [total_valid_pairs]
        valid_targets = negative_targets[negative_mask]  # [total_valid_pairs]
        
        if len(valid_logits) > 0:
            tcil_loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets)
        else:
            tcil_loss = torch.tensor(0.0, device=negative_text_embed.device)
            
        return tcil_loss

    def forward(self, outputs):
        if self.use_cls_loss:
            cls_eos_loss, cls_eos_acc = self.get_global_loss(
                outputs['eos_embed'], 
                outputs['cls_embed'], 
                outputs['logit_scale'])
        else:
            cls_eos_loss = 0.0
            cls_eos_acc = 0.0
        
        if self.use_mps_loss:
            mps_loss, mps_acc = self.get_mps_loss(
                outputs['mps_embed'],
                outputs['positive_mask'],
                outputs['cls_embed'],
                outputs['logit_scale']
            )
        else:
            mps_loss = 0.0
            mps_acc = 0.0
            
        if self.use_tcil_loss:
            text_embed  = outputs['positive_text_embed']                            # [B, K, D] or [B, 1, D]
            text_mask = outputs['positive_text_mask']                               # [B, K] or [B, 1]            
            positive_img_embed = outputs['positive_image_embed']                    # [B, K, D] or [B, 1, D]
            logit_scale_tc = outputs['logit_scale_tc']
            tcil_negatives_loss = torch.tensor(0.0, device=text_embed.device)
            
            if self.tcil_loss_mode == "k_positives_neg_ce":
                negative_text_embed = outputs.get('negative_text_embed')
                negative_img_embed = outputs.get('negative_image_embed')
                
                tcil_loss, tcil_acc = self.get_text_conditioned_image_loss(
                    text_embed,
                    text_mask,
                    positive_img_embed,
                    logit_scale_tc,
                    negative_text_embed=negative_text_embed,
                    negative_img_embed=negative_img_embed,
                )
            else:
                tcil_loss, tcil_acc = self.get_text_conditioned_image_loss(
                    text_embed, text_mask, positive_img_embed, logit_scale_tc
                )

                if self.use_negative_text_conditioning:
                    negative_text_embed = outputs['negative_text_embed'] # [B, (B-1)*(K+num_positives), D]
                    negative_img_embed = outputs['negative_image_embed'] # [B, (B-1)*(K+num_positives), D]
                
                    tcil_negatives_loss = self.get_tcil_negatives_only(
                        logit_scale_tc, negative_text_embed, negative_img_embed
                    )
                    tcil_loss = tcil_loss + self.tcil_neg_alpha * tcil_negatives_loss
        else:
            tcil_loss = 0.0
            tcil_acc = 0.0
            
        if self.use_tcil_loss:
            tci_diversity_metric_eos = torch.tensor(0.0, device=positive_img_embed.device)
            tci_diversity_metric_concepts = torch.tensor(0.0, device=positive_img_embed.device)
            tci_diversity_metric = torch.tensor(0.0, device=positive_img_embed.device)
        
            if self.use_tci_diversity_loss:
                if self.tci_div_separate:
                    assert self.num_eos_tokens > 0 and self.num_concept_tokens > 0, "num_eos_tokens and num_concept_tokens must be greater than 0"
                    if self.use_diversity_hinge:
                        # EOS: Stricter (they should be more diverse)
                        eos_div, tci_diversity_metric_eos = self.compute_diversity_hinge(
                            positive_img_embed[:, self.num_caption_tokens:self.num_eos_tokens, :], 
                            text_mask[:, self.num_caption_tokens:self.num_eos_tokens],
                            margin=0.7  # Penalize >0.9
                        )
                        
                        # Concepts: Relaxed (allow semantic overlap)
                        concept_div, tci_diversity_metric_concepts = self.compute_diversity_hinge(
                            positive_img_embed[:, self.num_eos_tokens:self.num_eos_tokens + self.num_concept_tokens, :],
                            text_mask[:, self.num_eos_tokens:self.num_eos_tokens + self.num_concept_tokens],
                            margin=0.7  # Penalize >0.9, allow <0.9
                        )
                        tci_diversity_loss = (eos_div + concept_div) / 2
                    else:
                        logit_scale_diversity_eos = outputs['logit_scale_tci_diversity_eos']
                        eos_div, tci_diversity_metric_eos = self.compute_tci_diversity_loss(
                                positive_img_embed[:, self.num_caption_tokens:self.num_eos_tokens, :], 
                                text_mask[:, self.num_caption_tokens:self.num_eos_tokens], 
                                logit_scale_diversity_eos
                            )
                            
                        logit_scale_diversity_concepts = outputs['logit_scale_tci_diversity_concepts']
                        concept_div, tci_diversity_metric_concepts = self.compute_tci_diversity_loss(
                            positive_img_embed[:, self.num_eos_tokens:self.num_eos_tokens + self.num_concept_tokens, :], 
                            text_mask[:, self.num_eos_tokens:self.num_eos_tokens + self.num_concept_tokens], 
                            logit_scale_diversity_concepts
                        )
                        tci_diversity_loss = (eos_div + concept_div) / 2                    
                else:
                    if self.use_diversity_hinge:
                        tci_diversity_loss, tci_diversity_metric = self.compute_diversity_hinge(
                            positive_img_embed[:, self.num_caption_tokens:, :], 
                            text_mask[:, self.num_caption_tokens:],
                            margin=0.7  # Penalize >0.7
                        )
                    else:
                        logit_scale_diversity = outputs['logit_scale_tci_diversity']
                        tci_diversity_loss, tci_diversity_metric = self.compute_tci_diversity_loss(
                            positive_img_embed[:, self.num_caption_tokens:, :], 
                            text_mask[:, self.num_caption_tokens:], 
                            logit_scale_diversity
                        )
            else:
                tci_diversity_loss = 0.0

                # EOS diversity
                if self.num_eos_tokens > 0:
                    tci_diversity_metric_eos = self.compute_diversity(
                        positive_img_embed[:, self.num_caption_tokens:self.num_eos_tokens, :], 
                        text_mask[:, self.num_caption_tokens:self.num_eos_tokens],
                    )
                
                if self.num_concept_tokens > 0:
                    tci_diversity_metric_concepts = self.compute_diversity(
                        positive_img_embed[:, self.num_eos_tokens:self.num_eos_tokens + self.num_concept_tokens, :],
                        text_mask[:, self.num_eos_tokens:self.num_eos_tokens + self.num_concept_tokens],
                    )
        else:
            tci_diversity_loss = 0.0

        
        total_loss = (self.cls_alpha * cls_eos_loss + 
                      self.tcil_alpha * tcil_loss + 
                      self.mps_alpha * mps_loss + 
                      self.tci_diversity_alpha * tci_diversity_loss)
        out = {
            'loss': total_loss,
        }
        
        if self.use_cls_loss:
            out['cls_eos_loss'] = self.cls_alpha * cls_eos_loss.detach()
            out['cls_eos_acc'] = cls_eos_acc.detach()
        if self.use_mps_loss:
            out['mps_loss'] = self.mps_alpha * mps_loss.detach()
            out['mps_acc'] = mps_acc.detach()
        if self.use_tcil_loss:
            out['text_conditioned_image_loss'] = self.tcil_alpha * tcil_loss.detach()
            out['text_conditioned_image_acc'] = tcil_acc.detach()
        
        if self.use_negative_text_conditioning:
            out['tcil_negatives_loss'] = self.tcil_neg_alpha * tcil_negatives_loss.detach()
            
        if self.use_tci_diversity_loss:
            out['tci_diversity_loss'] = self.tci_diversity_alpha * tci_diversity_loss.detach()
                    
        if self.use_tcil_loss:
            out['tci_diversity_eos'] = tci_diversity_metric_eos.detach()
            out['tci_diversity_concepts'] = tci_diversity_metric_concepts.detach()
            out['tci_diversity'] = tci_diversity_metric.detach()

        return out

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        image_embed = outputs['image_embed']
        text_embed = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = \
            utils.all_gather_batch([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'cls_eos_loss': loss, 'cls_eos_acc': acc}
