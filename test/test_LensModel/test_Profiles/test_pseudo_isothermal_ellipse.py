__author__ = "furcelay"


from lenstronomy.LensModel.Profiles.pseudo_isothermal_ellipse import (
    PseudoIsothermalEllipse,
)
import lenstronomy.Util.param_util as param_util

import numpy as np
import numpy.testing as npt
import pytest


class TestPIE(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.profile = PseudoIsothermalEllipse()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.0
        Rw = 0.5
        q, phi_G = 0.8, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        values = self.profile.function(
            x, y, sigma0, Rw, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(values[0], 1.3158500834489788, decimal=8)

        x = np.array([0])
        y = np.array([0])
        values = self.profile.function(
            x, y, sigma0, Rw, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(values[0], 0, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function(
            x, y, sigma0, Rw, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(values[0], 1.2227535214607006, decimal=8)
        npt.assert_almost_equal(values[1], 1.9408425265721638, decimal=8)
        npt.assert_almost_equal(values[2], 2.7288993387859795, decimal=8)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.0
        Rw = 0.5
        q, phi_G = 0.8, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.profile.derivatives(
            x, y, sigma0, Rw, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(f_x[0], 0.3176856549023636, decimal=8)
        npt.assert_almost_equal(f_y[0], 0.755029267897968, decimal=8)
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.profile.derivatives(
            x, y, sigma0, Rw, e1, e2, center_x=0, center_y=0
        )
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.profile.derivatives(
            x, y, sigma0, Rw, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(values[0][0], 0.3176856549023636, decimal=8)
        npt.assert_almost_equal(values[1][0], 0.755029267897968, decimal=8)
        npt.assert_almost_equal(values[0][1], 0.7604292311754575, decimal=8)
        npt.assert_almost_equal(values[1][1], 0.30103866299978294, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.0
        Rw = 0.5
        q, phi_G = 0.8, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(
            x, y, sigma0, Rw, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(f_xx[0], 0.2755010986081264, decimal=8)
        npt.assert_almost_equal(f_yy[0], 0.128642405896850943, decimal=8)
        npt.assert_almost_equal(f_xy[0], -0.10167140264626767, decimal=8)
        npt.assert_almost_equal(f_xy, f_yx, decimal=6)
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.profile.hessian(
            x, y, sigma0, Rw, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(values[0][0], 0.2755010986081264, decimal=8)
        npt.assert_almost_equal(values[3][0], 0.12864240589685094, decimal=8)

    def test_density_2d(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.0
        Rw = 0.5
        q, phi_G = 0.8, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        rho0 = self.profile.sigma2rho(sigma0, Rw)
        kappa = self.profile.density_2d(x, y, rho0, Rw, e1, e2, center_x=0, center_y=0)
        npt.assert_almost_equal(kappa, 0.20207175, decimal=8)


if __name__ == "__main__":
    pytest.main()
