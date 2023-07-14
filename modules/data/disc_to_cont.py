import numpy as np
import torch


def disc_to_cont(xp, fp):
    """
    Convert a discrete time-series to a continuous function via FFT.

    Args:
        xp (np.ndarray): 1D array of time-steps (uniformly spaced)
        fp (np.ndarray): 1D array of function evaluated at xp

    Returns:
        function(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor
    """
    N = len(fp)
    dx = xp[1] - xp[0]

    xi = np.fft.fftfreq(N, dx)  # frequencies
    fh = np.fft.fft(fp)  # fourier amplitudes

    cos_coeff = fh.real
    sin_coeff = fh.imag

    def continuous(x):
        """
        Continuous representation of the series passed to disc_to_cont.

        Args:
            x (np.ndarray | torch.Tensor): 1D array of points at which to evaluate the function.

        Returns:
            np.ndarray | torch.Tensor: function evaluated at x
        """

        numpy = isinstance(x, np.ndarray)
        pt = isinstance(x, torch.Tensor)
        assert numpy or pt

        # Reshape arrays for calculations
        x = x.reshape(1, -1)
        A = cos_coeff.reshape(-1, 1)
        B = sin_coeff.reshape(-1, 1)
        f = 2 * np.pi * xi.reshape(-1, 1)

        # Define functions
        sin = np.sin if numpy else torch.sin
        cos = np.cos if numpy else torch.cos
        if numpy:
            sum_ = lambda input: np.sum(input, axis=0)
        else:
            sum_ = lambda input: torch.sum(input, dim=0)
            A = torch.as_tensor(A)
            B = torch.as_tensor(B)
            f = torch.as_tensor(f)

        return sum_(A * cos(f * x) + B * sin(-f * x)) / len(f)

    return continuous
