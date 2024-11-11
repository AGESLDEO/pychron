# ===============================================================================
# Copyright 2012 Jake Ross
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

# ============= enthought library imports =======================
import logging
import string

from numpy import asarray, sqrt, sum, matrix, diagonal, array, exp, zeros, isfinite, abs, mean, ptp, polyfit, eye

# ============= standard library imports ========================
from scipy import optimize
from traits.api import Callable, List

from pychron.core.regression.base_regressor import BaseRegressor

logger = logging.getLogger("Regressor")

# ============= local library imports  ==========================


class FitError(BaseException):
    pass


class LeastSquaresRegressor(BaseRegressor):
    fitfunc = Callable
    initial_guess = List

    _covariance = None
    _nargs = 2

    def construct_fitfunc(self, fitstr):
        fitstr = fitstr.lstrip("custom:").lower().split("_")[0]
        import numexpr as ne

        def func(x, *args):
            ctx = dict(zip(string.ascii_lowercase[: len(args)][::-1], args))
            ctx["x"] = x
            return ne.evaluate(fitstr, local_dict=ctx)

        self._nargs = 2
        self.fitfunc = func

    def calculate(self, filtering=False):

        """
        Fit the model using curve_fit. If the fit produces a << c,
        perform constrained optimization instead.
        """
        # Define threshold for a << c
        threshold_ratio = 0.01  # Adjust this value as needed

        cxs = self.pre_clean_xs
        cys = self.pre_clean_ys

        if not self._check_integrity(cxs, cys):
            # logger.debug('A integrity check failed')
            # import traceback
            # traceback.print_stack()
            return

        if not filtering:
            # prevent infinite recursion
            fx, fy = self.calculate_filtered_data()
        else:
            fx, fy = cxs, cys
        try:
            initial_guess = self._calculate_initial_guess()
            valid = isfinite(fx) & isfinite(fy)
            print("fx array ", fx[valid])
            print("fy array ", fy[valid])
            coeffs, cov = optimize.curve_fit(
                self.fitfunc, fx[valid], fy[valid], p0=initial_guess
            )
            # This needs nan_policy="omit" but I have to update python to update scipy to 1.11
            print("fit function is ", self.fitfunc)
            print("coeffs are ", coeffs)
            self._coefficients = list(coeffs)
            self._covariance = cov
            self._coefficient_errors = list(sqrt(diagonal(cov)))
            print("coeff errors are ",self._coefficient_errors)

            # Check if a << c
            if self._coefficients[0] < threshold_ratio * self._coefficients[2]:
                raise ValueError("a is much smaller than c; switching to constrained optimization.")

        except ValueError:
            # Log the transition
            logger.warning("Trying a constrained optimization with a, b, c > 0.")

            # Define the objective function for constrained optimization
            def objective(params):
                return sum((self.ys - self.fitfunc(self.xs, *params)) ** 2)

            # Set bounds for constrained optimization
            bounds = [(0, None),  # a >= 0
                      (0, None),  # b >= 0
                      (0, None)]  # c >= 0

            # Perform constrained optimization
            result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            self._coefficients = list(result.x)

            # Need to estimate the covariance matrix separately after using minimize instead of curve_fit
            if hasattr(result, "hess_inv"):
                self._covariance = result.hess_inv.todense()  # Convert to dense matrix if sparse
            else:
                self._covariance = np.zeros((3, 3))  # Placeholder if Hessian is unavailable

        except RuntimeError:
            # Log the failure
            logger.warning("Exponential fit failed to converge. Falling back to linear fit.")
            self.fit = "linear"
            # Fallback to linear fit using numpy.polyfit
            try:
                linear_coeffs = polyfit(fx, fy, 1)  # Linear fit y = mx + b
                m, b = linear_coeffs
                self._coefficients = [b, m, 0]  # Map to equivalent structure [c, b, a]
                self._covariance = zeros((3, 3))  # Minimal covariance for linear fit
                self._coefficient_errors = [0, 0, 0]
            except Exception as e:
                # Final fallback to mean if linear fit also fails
                logger.error(f"Linear fit failed: {e}")
                self._coefficients = [mean(fy), 0, 0]
                self._covariance = eye(3)

    def _calculate_initial_guess(self):
        return zeros(self._nargs)

    def _calculate_coefficients(self):
        return self._coefficients

    def _calculate_coefficient_errors(self):
        return self._coefficient_errors

    def predict(self, x):
        return_single = False
        if not hasattr(x, "__iter__"):
            x = [x]
            return_single = True

        x = asarray(x)

        fx = self.fitfunc(x, *self._coefficients)
        if return_single:
            fx = fx[0]

        return fx

    def make_equation(self):
        return "A exp(-B*x) + C"

    # New mostly chatgpt version of predict_errors as I try to troubleshoot large error envelopes
    def predict_error(self, x, error_calc="sem"):
        """
        Calculate prediction error for an exponential fit.

        Args:
            x (array-like): Input values for which errors are predicted.
            error_calc (str): Type of error calculation ('sem' or 'sd').

        Returns:
            np.ndarray: Predicted errors for each input x.
        """
        return_single = False
        if not hasattr(x, "__iter__"):
            x = [x]
            return_single = True

        x = asarray(x)

        # Sensitivity matrix (Jacobian)
        sensitivity_matrix = zeros((len(x), len(self._coefficients)))
        sensitivity_matrix[:, 0] = exp(-self._coefficients[1] * x)  # ∂y/∂a
        sensitivity_matrix[:, 1] = -self._coefficients[0] * x * exp(
            -self._coefficients[1] * x
        )  # ∂y/∂b
        sensitivity_matrix[:, 2] = 1  # ∂y/∂c

        # Propagate the covariance matrix
        parameter_errors = sqrt(sum(sensitivity_matrix @ self._covariance * sensitivity_matrix, axis=1))

        # Calculate residual-based standard error
        residuals = self.ys - self.fitfunc(self.xs, *self._coefficients)
        standard_error = sqrt(sum(residuals ** 2) / (len(self.ys) - len(self._coefficients)))

        # Compute reduced chi-squared for regularization
        chi_squared_reduced = sum(residuals ** 2) / (len(self.ys) - len(self._coefficients))
        if chi_squared_reduced < 1:
            standard_error *= chi_squared_reduced  # Reduce influence of residual error

        # Combine parameter uncertainty with residual-based error
        combined_errors = sqrt(parameter_errors ** 2 + standard_error ** 2)

        if return_single:
            return combined_errors[0]
        return combined_errors

    # def predict_error(self, x, error_calc="sem"):
    #     """
    #     returns percent error
    #     """
    #     print("x is ", x)
    #     return_single = False
    #     if not hasattr(x, "__iter__"):
    #         x = [x]
    #         return_single = True
    #
    #     sef = self.calculate_standard_error_fit()
    #     r, _ = self._covariance.shape
    #
    #     def calc_error(xi):
    #         Xk = matrix(
    #             [
    #                 xi,
    #             ]
    #             * r
    #         ).T
    #
    #         varY_hat = Xk.T * self._covariance * Xk
    #         if error_calc == "sem":
    #             se = sef * sqrt(varY_hat)
    #         else:
    #             se = sqrt(sef**2 + sef**2 * varY_hat)
    #         print("error is ", se[0, 0])
    #         return se[0, 0]
    #
    #     fx = array([calc_error(xi) for xi in x])
    #     # fx = ys * fx / 100.
    #     print("error array is ", fx)
    #     if return_single:
    #         fx = fx[0]
    #     print("return single is ",return_single)
    #     return fx


class ExponentialRegressor(LeastSquaresRegressor):
    def __init__(self, *args, **kw):
        def fitfunc(x, a, b, c):
            return a * exp(-b * x) + c

        self.fitfunc = fitfunc
        self.fit = "exponential"
        super(ExponentialRegressor, self).__init__(*args, **kw)

    def _calculate_initial_guess(self):

        # Trying to make it more robust with chatgpt help due to error envelopes blowing up
        # Estimate the baseline (C) as the minimum or maximum value of y
        if len(self.ys) > 3:
            ig_c = mean(self.ys[-3:])  # Average of the last few points (assumes steady behavior)
        else:
            ig_c = self.ys[-1] # If we don't have at least four points, just use the last point

        # Estimate the amplitude (A) as the difference between the first point and the baseline
        ig_a = self.ys[0] - ig_c

        # Estimate the decay/growth rate (B) based on the trend of y
        # Use a log ratio to estimate the rate of change over the range of x
        if len(self.xs) > 1:
            ig_b = abs(ig_a) / (ptp(self.xs))  # Default to a positive rate
        else:
            ig_b = 0.1  # Default to a small positive rate for small datasets

        # Handle edge cases where the data does not vary
        if ig_a == 0:
            ig_b = 0.1

        # my first refinement
        # ig_c = self.ys[-1]
        # ig_a = self.ys[0] - ig_c
        # if ig_c > ig_a:
        #     ig_b = 1e-2
        # else:
        #     ig_b = -1e-2
        ig = ig_a, ig_b, ig_c
        # Jake's version
        # if self.ys[0] > self.ys[-1]:
        #     ig = 1e-3, -1e-2, 1e-3
        # else:
        #     ig = -1e-3, 1e-2, 1e-3
        return ig


# ============= EOF =============================================
