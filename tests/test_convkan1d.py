import pytest
import torch
from torch.nn import Conv1d
from convkan.convkan1d_layer import ConvKAN1D

@pytest.fixture
def sample_input_1d():
    return torch.randn(2, 3, 32, dtype=torch.float32)

@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding,groups,dilation",
    [
        (3, 16, 3, 1, 1, 1, 2),
        (3, 32, 5, 2, 1, 1, 1),
        (3, 18, 5, 1, 2, 1, 2),
        (3, 18, 5, 1, 1, 3, 1),
        (3, 18, 5, 2, 2, 3, 2),
        (3, 16, 5, 1, "same", 1, 2),
        (3, 16, 5, 2, "valid", 1, 2),
    ],
)
def test_conv_kan1d_forward_shape(
        sample_input_1d, in_channels, out_channels, kernel_size, stride, padding, groups, dilation
):
    model = ConvKAN1D(
        in_channels, out_channels, kernel_size, stride, padding, groups=groups, dilation=dilation
    )

    torch_model = Conv1d(
        in_channels, out_channels, kernel_size, stride, padding, groups=groups, dilation=dilation
    )

    out1 = model(sample_input_1d)
    out2 = torch_model(sample_input_1d)

    assert out1.shape == out2.shape, f"Output shape is incorrect: {out1.shape} vs {out2.shape}"

def test_invalid_groups_1d():
    with pytest.raises(ValueError):
        ConvKAN1D(3, 16, 3, groups=2)

@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate"])
def test_padding_modes_1d(sample_input_1d, padding_mode):
    kernel_size = 3
    stride = 1
    padding = 1
    in_channels = 3
    out_channels = 16

    model = ConvKAN1D(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
    )
    output = model(sample_input_1d)

    input_length = sample_input_1d.shape[2]
    expected_length = (input_length + 2 * padding - kernel_size) // stride + 1

    assert output.shape == (
        sample_input_1d.shape[0],
        out_channels,
        expected_length,
    ), (
        f"Output shape is incorrect for padding mode {padding_mode}. "
        f"Expected ({sample_input_1d.shape[0], out_channels, expected_length}), got {output.shape}"
    )

@pytest.mark.parametrize(
    "in_channels, out_channels, groups, kernel_size",
    [(4, 8, 2, 3), (6, 12, 3, 3)],
)
def test_group_effect_on_output_1d(in_channels, out_channels, groups, kernel_size):
    model = ConvKAN1D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
    )

    base_input = torch.rand(1, in_channels, 10)
    modified_input = base_input.clone()
    group_size = in_channels // groups
    group_to_modify = 1
    modified_input[
    :, group_size * group_to_modify: group_size * (group_to_modify + 1)
    ] *= 2

    output_base = model(base_input)
    output_modified = model(modified_input)

    for g in range(groups):
        start_channel = g * (out_channels // groups)
        end_channel = start_channel + (out_channels // groups)

        if g == group_to_modify:
            assert not torch.allclose(
                output_base[:, start_channel:end_channel],
                output_modified[:, start_channel:end_channel],
                atol=1e-6,
            ), f"Output channels {start_channel}-{end_channel} should be different due to input modification."
        else:
            assert torch.allclose(
                output_base[:, start_channel:end_channel],
                output_modified[:, start_channel:end_channel],
                atol=1e-6,
            ), f"Output channels {start_channel}-{end_channel} should be unchanged."

def test_gradient_flow_1d(sample_input_1d):
    model = ConvKAN1D(3, 16, 3, stride=1, padding=1)
    model.train()
    output = model(sample_input_1d)
    loss = output.sum()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None, "Gradients should not be None."

def test_dtype_handling_1d():
    model = ConvKAN1D(3, 16, 3)
    inputs = torch.randn(2, 3, 32, dtype=torch.double)
    model.double()
    output = model(inputs)
    assert (
            output.dtype == torch.double
    ), "Output dtype should match input dtype (double)."
