#include <cmath>
#include <fstream>
#include <iostream>

#include <armadillo>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>

const double mass = 939.0 / 197.0;
// parameters f o r potential ,h bar ∗c = 1
const int a = 1;
const int b = 4;
const int c = 7;
const double miu = 0.7;
const double Va = -10.463 / 197.0;
const double Vb = -1650.6 / 197.0;
const double Vc = 6484.3 / 197.0;
const double ZERO = 1.0E-10;

using namespace arma;
using namespace std;

double Poten(double k1, double k2);
void GaussLegendreQuadrature(double, double, double *, double *, int);

int main() {
  int n = 100;
  // set up the mesh points and weights
  double *x = new double[n + 1];
  double *w = new double[n + 1];
  double *r = new double[n + 1];
  double *s = new double[n + 1];
  // set up matrices V, A and R
  mat A = zeros<mat>(n + 1, n + 1);
  mat R = zeros<mat>(n + 1, n + 1);
  mat V = zeros<mat>(n + 1, n + 1);
  mat u = zeros<vec>(n + 1);
  GaussLegendreQuadrature(-1.0, 1.0, x, w, n);
  const double pi = 3.14159265359;

  for (int i = 0; i < n; i++) {
    double xx = 0.25 * pi * (x[i] + 1.0);
    r[i] = tan(xx); // K_i = tan(π/4(1+x_i))
    // ω_i = π/4 * ...
    s[i] = 0.25 * pi / (cos(xx) * cos(xx)) * w[i];
  }
  // start loop for different k0
  for (int np = 0; np < n; ++np) {
    r[n] = np * 0.03 + 0.01;
    std::cout << n << " = " << r[n]<< std::endl;
    for (int i = 0; i < n; ++i) {
      u(i) = 2.0 / pi * s[i] * r[i] * r[i] * mass / (r[n] * r[n] - r[i] * r[i]);
    }
    u(n) = 0;
    for (int i = 0; i < n; ++i) {
      u(n) = u(n) - 2.0 / pi * mass * s[i] / (r[n] * r[n] - r[i] * r[i]);
    }
    u(n) = u(n) * r[n] * r[n];

    for (int i = 0; i < n + 1; ++i) {
      for (int j = 0; j < n + 1; ++j) {
          V(i, j) = Poten(r[i], r[j]);
          if (i != j) {
            A(i, j) = -V(i, j) * u(j);
        } else {
          A(i, j) = 1.0 - V(i, j) * u(j);
        }
      }
    }
    mat A_inv = inv(A);
    R = A_inv * V;
    double sigma = atan(-R(n, n) * mass * r[n]);
    double energy = r[n] * r[n] / mass * 197;
    cout << sigma / pi * 180 << " " << energy << endl;
  }
  delete[] x;
  delete[] w;
  delete[] s;
  delete[] r;
  return 0;
}

// potential in momentum space
double Poten(double k1, double k2) {
  double value = 0;
  value += 0.25 * Va / (miu * k1 * k2) *
           log((pow(k1 + k2, 2) + pow(miu * a, 2)) /
               (pow(k1 - k2, 2) + pow(miu * a, 2)));
  value += 0.25 * Vb / (miu * k1 * k2) *
           log((pow(k1 + k2, 2) + pow(miu * b, 2)) /
               (pow(k1 - k2, 2) + pow(miu * b, 2)));
  value += 0.25 * Vc / (miu * k1 * k2) *
           log((pow(k1 + k2, 2) + pow(miu * c, 2)) /
               (pow(k1 - k2, 2) + pow(miu * c, 2)));
  return value;
}

void GaussLegendreQuadrature(double x1, double x2, double x[], double w[],
                             int n) {
  int m, j, i;
  double z1, z, xm, xl, pp, p2, p3, p1;
  const double pi = 3.14150265359;
  double *x_low, *x_high, *w_low, *w_high;

  m = (n + 1) / 2;
  xm = 0.5 * (x2 + x1);
  xl = 0.5 * (x2 - x1);
  x_low = x;
  x_high = x + n - 1;
  w_low = w;
  w_high = w + n - 1;

  for (i = 1; i <= m; i++) {
    z = cos(pi * (i - 0.25) / (n + 0.5));
    do {
      p1 = 1;
      p2 = 0;
      for (j = 1; j <= n; j++) {
        p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }

      pp = n * (z * p1 - p2) / (z * z - 1);
      z1 = z;
      z = z1 - p1 / pp;
    } while (fabs(z - z1) > ZERO);
    *(x_low++) = xm - xl * z;
    *(x_high--) = xm + xl * z;
    *w_low = 2 * xl / ((1.0 - z * z) * pp * pp);
    *(w_high--) = *(w_low++);
  }
}
