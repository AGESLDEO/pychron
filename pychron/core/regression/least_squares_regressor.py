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

import numpy as np

# ============= standard library imports ========================
from scipy import optimize
from traits.api import Callable, List

from pychron.core.regression.base_regressor import BaseRegressor

logger = logging.getLogger("Regressor")

# ============= local library imports  ==========================


class FitError(BaseException):
    pass


def _calculate_linear_fit(fx, fy):

    # Design matrix
    matX = np.vstack([fx, np.ones_like(fx)]).T

    # Perform the linear fit
    beta, _, _, _ = np.linalg.lstsq(matX, fy, rcond=None)  # [slope, intercept]
    slope, intercept = beta

    # Compute residuals
    residuals = fy - (slope * fx + intercept)
    residual_variance = np.var(residuals, ddof=2)

    # Compute covariance matrix
    covariance_matrix = residual_variance * np.linalg.inv(matX.T @ matX)

    return slope, intercept, covariance_matrix

def _calculate_average(fy):

    mean_y = np.mean(fy)
    sem = np.std(fy) / np.sqrt(len(fy))
    covariance_matrix = np.diag([sem ** 2, 0.0, sem ** 2])

    return mean_y, covariance_matrix, sem


class LeastSquaresRegressor(BaseRegressor):
    fitfunc = Callable
    initial_guess = List

    _covariance = None
    _nargs = 2

    def construct_fitfunc(self, fitstr):
        fitstr = fitstr.lstrip("custom:").lower().split("_")[0]

        def func(x, *args):
            ctx = dict(zip(string.ascii_lowercase[: len(args)][::-1], args))
            ctx["x"] = x
            return np.ne.evaluate(fitstr, local_dict=ctx)

        self._nargs = 2
        self.fitfunc = func

    def calculate(self, filtering=False):

        """
        Fit the model using curve_fit. If the fit produces a << c,
        perform constrained optimization instead.
        """
        # Define threshold for a << c
        threshold = 10  # Adjust this value as needed

        cxs = self.pre_clean_xs
        cys = self.pre_clean_ys

        if not self._check_integrity(cxs, cys):
            # logger.debug('A integrity check failed')
            # import traceback
            # traceback.print_stack()
            return

        if not filtering:
            # prevent infinite recursion
            self.fx, self.fy = self.calculate_filtered_data()
        else:
            self.fx, self.fy = cxs, cys
        try:
            initial_guess = self._calculate_initial_guess()
            valid = np.isfinite(self.fx) & np.isfinite(self.fy)
            self.fx = self.fx[valid]
            self.fy = self.fy[valid]
            coeffs, cov = optimize.curve_fit(
                self.fitfunc, self.fx, self.fy, p0=initial_guess
            )
            # This needs nan_policy="omit" but I have to update python to update scipy to 1.11 so did it manually
            self._coefficients = list(coeffs)
            self._covariance = cov
            self._coefficient_errors = list(np.sqrt(np.diagonal(cov)))
            print("coeff errors are ",self._coefficient_errors)
            if not self._covariance_matrix_test(self.fx, self.fy):
                return

            # Check if a << c
            if self._coefficients[0] < -1 * threshold * abs(self._coefficients[2]):
                raise ValueError("a is much smaller than c; switching to constrained optimization.")

            # Check if errors are reasonable
            if any(np.asarray(self._coefficient_errors)/np.asarray(self._coefficients) > 1):
                raise ValueError("coefficient errors are unreasonable; switching to constrained optimization.")

        except ValueError:
            # Log the transition
            logger.warning("Trying a constrained optimization with a, b, c > 0.")

            # Define the objective function for constrained optimization
            def objective(params):
                return np.sum((self.fy - self.fitfunc(self.fx, *params)) ** 2)

            # Set bounds for constrained optimization
            bounds = [(0, None),  # a >= 0
                      (None, None),  # b >= 0
                      (0, None)]  # c >= 0

            # Perform constrained optimization
            result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            self._coefficients = list(result.x)

            # Need to estimate the covariance matrix separately after using minimize instead of curve_fit
            if hasattr(result, "hess_inv"):
                self._covariance = result.hess_inv.todense()  # Convert to dense matrix if sparse
            else:
                self._covariance = np.zeros((3, 3))  # Placeholder if Hessian is unavailable

            self._coefficient_errors = list(np.sqrt(np.diagonal(cov)))

            if not self._covariance_matrix_test(self.fx, self.fy):
                return

        except RuntimeError:
            # Log the failure
            logger.warning("Exponential fit failed to converge. Falling back to linear fit.")
            self._try_fallbacks(self.fx, self.fy)

    def _calculate_initial_guess(self):
        return np.zeros(self._nargs)

    def _calculate_coefficients(self):
        return self._coefficients

    def _calculate_coefficient_errors(self):
        return self._coefficient_errors

    def _try_fallbacks(self, fx, fy):
        try:
            print("Attempting linear fit fallback...")
            # Fallback to linear fit using numpy.polyfit
            self.fit = "linear"
            [slope, intercept, covariance_matrix] = _calculate_linear_fit(fx, fy)

            # Assign results
            print(f"Linear fit succeeded: slope={slope}, intercept={intercept}")
            self._coefficients = [slope, 0.0, intercept]
            self._covariance = covariance_matrix
            self._coefficient_errors = [np.sqrt(covariance_matrix[0, 0]), np.sqrt(covariance_matrix[1, 1])]

        except Exception as e:
            print(f"Linear fit failed with error: {e}. Falling back to average.")
            self.fit = "average"
            [mean_y, covariance_matrix, sem] = _calculate_average(fy)

            # Assign results
            print(f"Average fallback: mean_y={mean_y}, sem={sem}")
            self._coefficients = [0.0, 0.0, mean_y]
            self._covariance = covariance_matrix
            self._coefficient_errors = [0.0, 0.0, sem]

    def _covariance_matrix_test(self, fx, fy):
        """
        Test the validity of the covariance matrix and handle fallback to an average fit if needed.

        Parameters:
            fy (array): The dependent variable data used in the fit.

        Returns:
            bool: True if the covariance matrix is valid, False if it failed and the fallback was applied.
        """
        if self._covariance is not None:
            diagonal_elements = np.diag(self._covariance)
            if np.any(diagonal_elements > 1e6) or np.any(diagonal_elements < 0) or any(np.asarray(self._coefficient_errors)/np.asarray(self._coefficients) > 1):
                print("Covariance matrix indicates unreliable fit. Falling back to linear.")
                self.fit = "linear"
                self._try_fallbacks(fx, fy)
                # [mean_y, covariance_matrix, sem] = _calculate_average(fy)
                #
                # # Assign results
                # self._coefficients = [0.0, 0.0, mean_y]
                # self._covariance = covariance_matrix
                # self._coefficient_errors = [0.0, 0.0, sem]
                return False  # Covariance matrix failed the test
            return True  # Covariance matrix passed the test
        else:
            return False

    def predict(self, x):
        return_single = False
        if not hasattr(x, "__iter__"):
            x = [x]
            return_single = True

        x = np.asarray(x)

        if self.fit == "exponential":
            # Use the exponential fit function
            fx = self.fitfunc(x, *self._coefficients)
        elif self.fit == "linear":
            # Linear fit: y = slope * x + intercept
            slope, intercept = self._coefficients[:2]
            fx = slope * x + intercept
        else:
            # Fallback to the average
            fx = np.full_like(x, self._coefficients[0], dtype=float)

        if return_single:
            fx = fx[0]

        return fx

    def make_equation(self):
        return "A exp(-B*x) + C"

    # New mostly chatgpt version of predict_errors as I try to troubleshoot large error envelopes
    def predict_error(self, x, error_calc="sem"):
        """
        Calculate prediction error for the current fit.

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

        x = np.asarray(x)

        # Construct the sensitivity matrix based on the fit type
        if self.fit == "exponential":
            sensitivity_matrix = np.zeros((len(x), 3))
            sensitivity_matrix[:, 0] = np.exp(-self._coefficients[1] * x)  # ∂y/∂a
            sensitivity_matrix[:, 1] = -self._coefficients[0] * x * np.exp(
                -self._coefficients[1] * x
            )  # ∂y/∂b
            sensitivity_matrix[:, 2] = 1  # ∂y/∂c

            # Propagate the covariance matrix
            parameter_errors = np.sqrt(
                np.sum(sensitivity_matrix @ self._covariance * sensitivity_matrix, axis=1)
            )

            # Calculate residuals only for exponential fits
            residuals = self.ys - self.fitfunc(self.xs, *self._coefficients)
            standard_error = np.sqrt(np.sum(residuals ** 2) / (len(self.ys) - len(self._coefficients)))

        elif self.fit == "linear":
            sensitivity_matrix = np.zeros((len(x), 2))
            sensitivity_matrix[:, 0] = x  # ∂y/∂slope
            sensitivity_matrix[:, 1] = 1  # ∂y/∂intercept

            # Propagate the covariance matrix
            parameter_errors = np.sqrt(
                np.sum(sensitivity_matrix @ self._covariance * sensitivity_matrix, axis=1)
            )
            standard_error = 0  # Linear fits typically do not calculate residual error here

        else:  # Average fit
            # Standard error of the mean (SEM) for average fit
            sem = np.std(self.ys) / np.sqrt(len(self.ys))
            parameter_errors = np.full_like(x, sem, dtype=float)  # SEM as the error
            standard_error = sem

        # Combine parameter uncertainty with residual-based error
        combined_errors = np.sqrt(parameter_errors ** 2 + standard_error ** 2)

        if return_single:
            return combined_errors[0]
        return combined_errors

class ExponentialRegressor(LeastSquaresRegressor):
    def __init__(self, *args, **kw):
        def fitfunc(x, a, b, c):
            return a * np.exp(-b * x) + c

        self.fitfunc = fitfunc
        self.fit = "exponential"
        super(ExponentialRegressor, self).__init__(*args, **kw)

    def _calculate_initial_guess(self):

        # Trying to make it more robust with chatgpt help due to error envelopes blowing up
        # Estimate the baseline (C) as the minimum or maximum value of y
        if len(self.ys) > 3:
            ig_c = np.mean(self.ys[-3:])  # Average of the last few points (assumes steady behavior)
        else:
            ig_c = self.ys[-1] # If we don't have at least four points, just use the last point

        # Estimate the amplitude (A) as the difference between the first point and the baseline
        ig_a = self.ys[0] - ig_c

        # Estimate the decay/growth rate (B) based on half-max behavior
        # Use a log ratio to estimate the rate of change over the range of x
        half_max = ig_a/2 + ig_c
        decay_index = np.argmax(self.ys < half_max)
        if len(self.ys) > 5 and decay_index > 0:
            ig_b = np.log(2) / (self.xs[decay_index] - self.xs[0])
        else:
            ig_b = 0.1  # Default to a small positive rate

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
