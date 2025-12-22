"""Utility functions for tensor and dictionary-of-tensor operations."""

from typing import Any, Dict, Sequence, Union

import torch


def select_entries(
    tensor_input: Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    indices: Union[int, Sequence[int], torch.Tensor],
) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
    """Select specific entries from a tensor or dictionary containing tensors or nested tensor dictionaries.

    This function is useful for selecting specific indices from batched tensors or tensor dictionaries,
    such as selecting specific environments from a batch of states.

    Args:
        tensor_input: Can be:
                     - A single tensor
                     - Dictionary of tensors: {key: tensor}
                     - Nested dictionary of tensors: {key: {tensor_name: tensor}}
        indices: Integer, sequence of integers, or tensor of indices to select.
                Can be a single int, list[int], or torch.Tensor of indices.

    Returns:
        - If input is a tensor: returns indexed tensor
        - If input is a dict: returns dict with same structure, but with tensors indexed
          to only include the selected entries.

    Example:
        >>> # Single tensor
        >>> tensor = torch.randn(64, 12)
        >>> selected = select_entries(tensor, [0, 5, 10])
        >>> # Returns tensor for indices 0, 5, and 10 only

        >>> # Dictionary of tensors
        >>> tensor_dict = {
        ...     "joint_q": torch.randn(64, 12),
        ...     "states": {"joint_qd": torch.randn(64, 12), "body_q": torch.randn(64, 13)},
        ... }
        >>> selected = select_entries(tensor_dict, [0, 5, 10])
        >>> # Returns dict with tensors for indices 0, 5, and 10 only
    """
    if isinstance(tensor_input, dict):
        # If input is a dictionary, recursively apply to each value
        # Convert indices to tensor if needed
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, (list, tuple)):
            # Try to get device from first tensor value
            device = None
            for value in tensor_input.values():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break
                elif isinstance(value, dict):
                    for v in value.values():
                        if isinstance(v, torch.Tensor):
                            device = v.device
                            break
                    if device is not None:
                        break
            indices = torch.tensor(indices, dtype=torch.long, device=device)

        result = {}
        for key, value in tensor_input.items():
            result[key] = select_entries(value, indices)
        return result
    elif isinstance(tensor_input, torch.Tensor):
        # If input is a tensor, index it directly
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, (list, tuple)):
            indices = torch.tensor(indices, dtype=torch.long, device=tensor_input.device)
        return tensor_input[indices]
    else:
        # For non-tensor values, return as-is
        return tensor_input


def duplicate_entries(
    tensor_input: Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]], num_copies: int
) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
    """Duplicate entries along the leading dimension of a tensor or dictionary of tensors.

    Each entry in the leading dimension is repeated `num_copies` times.
    For example, [1, 2] with num_copies=2 becomes [1, 1, 2, 2].

    Args:
        tensor_input: Can be:
                     - A single tensor
                     - Dictionary of tensors: {key: tensor}
                     - Nested dictionary of tensors: {key: {tensor_name: tensor}}
        num_copies: Number of times to duplicate each entry along the leading dimension.

    Returns:
        - If input is a tensor: returns duplicated tensor
        - If input is a dict: returns dict with same structure, but with tensors duplicated along dim=0.
        Original shape [N, ...] becomes [N * num_copies, ...].

    Example:
        >>> # Single tensor
        >>> tensor = torch.tensor([[1.0], [2.0]])  # shape: [2, 1]
        >>> duplicated = duplicate_entries(tensor, num_copies=2)
        >>> # Returns: tensor([[1.0], [1.0], [2.0], [2.0]])  # shape: [4, 1]

        >>> # Dictionary of tensors
        >>> tensor_dict = {
        ...     "joint_q": torch.tensor([[1.0], [2.0]]),  # shape: [2, 1]
        ...     "states": {
        ...         "body_q": torch.tensor([[3.0], [4.0]])  # shape: [2, 1]
        ...     },
        ... }
        >>> duplicated = duplicate_entries(tensor_dict, num_copies=2)
        >>> # Returns:
        >>> # {"joint_q": tensor([[1.0], [1.0], [2.0], [2.0]]),  # shape: [4, 1]
        >>> #  "states": {"body_q": tensor([[3.0], [3.0], [4.0], [4.0]])}  # shape: [4, 1]
    """
    if isinstance(tensor_input, dict):
        # If input is a dictionary, recursively apply to each value
        result = {}
        for key, value in tensor_input.items():
            result[key] = duplicate_entries(value, num_copies)
        return result
    elif isinstance(tensor_input, torch.Tensor):
        # If input is a tensor, duplicate it along dim=0
        if num_copies < 1:
            raise ValueError(f"num_copies must be >= 1, got {num_copies}")
        return tensor_input.repeat_interleave(num_copies, dim=0)
    else:
        # For non-tensor values, return as-is
        return tensor_input


def check_groups_same(
    tensor_input: Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
    group_size: int,
    dim: int = 0,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> Union[bool, Dict[str, Union[bool, Dict[str, bool]]]]:
    """Check if all entries within each group are identical for a tensor or dictionary of tensors.

    This function applies the check recursively to handle:
    - Single tensors
    - Dictionaries of tensors: {key: tensor}
    - Nested dictionaries of tensors: {key: {tensor_name: tensor}}

    This is useful for verifying that duplicate_entries worked correctly on
    complex state dictionaries.

    Args:
        tensor_input: Can be:
                     - A single tensor
                     - Dictionary of tensors: {key: tensor}
                     - Nested dictionary of tensors: {key: {tensor_name: tensor}}
        group_size: Size of each group to check (should match num_copies from duplicate_entries).
        dim: Dimension along which to group elements. Default is 0 (leading dimension).
        atol: Absolute tolerance for floating-point comparison. Default is 1e-8.
              Set to 0.0 to use only relative tolerance.
        rtol: Relative tolerance for floating-point comparison. Default is 1e-5.
              To use only relative tolerance, set atol=0.0 and provide rtol value (e.g., rtol=1e-5).

    Returns:
        - If input is a tensor: returns a single bool
        - If input is a dict of tensors: returns {key: bool}
        - If input is nested dict: returns {key: {tensor_name: bool}}

    Example:
        >>> # Single tensor
        >>> tensor = torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4]])
        >>> check_groups_same(tensor, group_size=2, dim=0)
        True

        >>> # Dictionary of tensors
        >>> tensor_dict = {
        ...     "joint_q": torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4]]),
        ...     "body_q": torch.tensor([[5, 6], [5, 6], [7, 8], [7, 8]]),
        ... }
        >>> result = check_groups_same(tensor_dict, group_size=2, dim=0)
        >>> # Returns {"joint_q": True, "body_q": True}

        >>> # Nested dictionary of tensors
        >>> nested_dict = {
        ...     "robot_states": {
        ...         "joint_q": torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4]]),
        ...         "body_q": torch.tensor([[5, 6], [5, 6], [7, 8], [7, 8]]),
        ...     },
        ...     "progress_buf": torch.tensor([0, 0, 1, 1]),
        ... }
        >>> result = check_groups_same(nested_dict, group_size=2, dim=0)
        >>> # Returns {"robot_states": {"joint_q": True, "body_q": True}, "progress_buf": True}
    """
    if isinstance(tensor_input, dict):
        # If input is a dictionary, recursively apply to each value
        result = {}
        for key, value in tensor_input.items():
            result[key] = check_groups_same(value, group_size, dim, atol, rtol)
        return result
    elif isinstance(tensor_input, torch.Tensor):
        # If input is a tensor, apply the check directly
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")

        tensor_size = tensor_input.shape[dim]
        if tensor_size % group_size != 0:
            raise ValueError(
                f"Tensor size along dim {dim} ({tensor_size}) must be divisible by group_size ({group_size})"
            )

        num_groups = tensor_size // group_size

        # Reshape tensor to group the elements
        # Move the target dimension to the front, reshape, then check
        tensor_permuted = tensor_input.transpose(0, dim)
        # Reshape: [tensor_size, ...] -> [num_groups, group_size, ...]
        tensor_reshaped = tensor_permuted.reshape(num_groups, group_size, *tensor_permuted.shape[1:])

        # Check if all entries within each group are identical
        # For each group, check if all entries equal the first entry
        first_entries = tensor_reshaped[:, 0:1, ...]  # [num_groups, 1, ...]
        group_entries = tensor_reshaped  # [num_groups, group_size, ...]

        # Expand first_entries to match group_entries shape for comparison
        first_entries_expanded = first_entries.expand_as(group_entries)

        # Check if all entries in each group are close to the first entry (within tolerance)
        # Use torch.allclose for floating-point comparison with tolerance
        if tensor_input.is_floating_point():
            # For floating-point tensors, use allclose with tolerance
            # allclose returns a single boolean, so we can return it directly
            all_same = torch.allclose(group_entries, first_entries_expanded, atol=atol, rtol=rtol)
            return all_same
        else:
            # For integer tensors, use exact equality
            all_same = (group_entries == first_entries_expanded).all()
            return all_same.item() if all_same.numel() == 1 else all_same.all().item()
    else:
        # For non-tensor values, return as-is (or could raise an error)
        return tensor_input


def all_dict_values_true(bool_dict: Union[bool, Dict[str, Union[bool, Dict[str, Any]]]]) -> bool:
    """Check if all boolean values in a dictionary (including nested dictionaries) are True.

    This function recursively traverses a dictionary structure and checks that all
    boolean values are True. Useful for asserting on results from functions like
    check_groups_same that return nested dictionaries of booleans.

    Args:
        bool_dict: Can be:
                  - A single boolean
                  - Dictionary of booleans: {key: bool}
                  - Nested dictionary of booleans: {key: {nested_key: bool}}

    Returns:
        True if all boolean values are True, False otherwise.

    Example:
        >>> # Single boolean
        >>> all_dict_values_true(True)
        True

        >>> # Dictionary of booleans
        >>> result = {"joint_q": True, "body_q": True}
        >>> all_dict_values_true(result)
        True

        >>> # Nested dictionary
        >>> result = {"robot_states": {"joint_q": True, "body_q": True}, "progress_buf": True}
        >>> all_dict_values_true(result)
        True

        >>> # Returns False if any value is False
        >>> result = {"robot_states": {"joint_q": True, "body_q": False}, "progress_buf": True}
        >>> all_dict_values_true(result)
        False
    """
    if isinstance(bool_dict, dict):
        # Recursively check all values in the dictionary
        return all(all_dict_values_true(value) for value in bool_dict.values())
    elif isinstance(bool_dict, bool):
        # Base case: return the boolean value
        return bool_dict
    else:
        # For non-boolean values, treat as True (or could raise an error)
        return True
