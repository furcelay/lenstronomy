__author__ = "furcelay"


from lenstronomy.LensModel.Profiles.jaffe_ellipse import (
    JaffeEllipse,
)
import lenstronomy.Util.param_util as param_util

import numpy as np
import numpy.testing as npt
import pytest


class TestPJAFFESpherical(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.profile = JaffeEllipse()
        self.spherical = self.profile._spherical

    def test_function(self):
        # test that the gradient of the potential is consistent with the deflection
        x = np.linspace(-1, 1, 30)
        y = np.linspace(-1, 1, 30)
        xx, yy = np.meshgrid(x, y)
        dx = dy = x[1] - x[0]
        theta_E = 1.0
        Ra, Rs = 0.5, 0.8
        # theta_E = sigma0 * 2 * Ra * Rs**2 / (Rs**2 - Ra**2)
        sigma0 = theta_E / (2 * Ra)
        q, phi_G = 0.7, 0.3
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        potential = self.profile.function(xx, yy, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0)
        potential_dy, potential_dx = np.gradient(potential, dy, dx)
        f_x, f_y = self.profile.derivatives(
            xx, yy, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(potential_dx, f_x, decimal=2)
        npt.assert_almost_equal(potential_dy, f_y, decimal=2)

    def test_derivatives(self):
        theta_E = 1.0
        Ra, Rs = 0.5, 0.8
        # theta_E = sigma0 * 2 * Ra * Rs**2 / (Rs**2 - Ra**2)
        sigma0 = theta_E  / (2 * Ra)
        q, phi_G = 0.999, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)

        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.profile.derivatives(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        print(f_x, f_y)
        npt.assert_almost_equal(f_x, 0, decimal=5)
        npt.assert_almost_equal(f_y, 0, decimal=5)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = self.profile.derivatives(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        f_x_shp, f_y_shp = self.spherical.derivatives(
            x, y, sigma0, Ra, Rs, center_x=0, center_y=0
        )
        npt.assert_almost_equal(f_x, f_x_shp, decimal=3)
        npt.assert_almost_equal(f_y, f_y_shp, decimal=3)

    def test_hessian(self):
        theta_E = 1.0
        Ra, Rs = 0.5, 0.8
        sigma0 = theta_E / (2 * Ra)
        q, phi_G = 0.999, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        f_xx_sph, f_xy_sph, f_yx_sph, f_yy_sph = self.spherical.hessian(
            x, y, sigma0, Ra, Rs, center_x=0, center_y=0
        )
        npt.assert_almost_equal(f_xx, f_xx_sph, decimal=3)
        npt.assert_almost_equal(f_xy, f_xy_sph, decimal=3)
        npt.assert_almost_equal(f_yx, f_yx_sph, decimal=3)
        npt.assert_almost_equal(f_yy, f_yy_sph, decimal=3)

if __name__ == "__main__":
    pytest.main()
