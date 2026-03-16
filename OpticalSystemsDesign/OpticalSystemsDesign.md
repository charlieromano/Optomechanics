# Optical Systems Design



### Geometrical optics

$$
m = -\frac{s_i}{s_0}
$$

$$
\frac{1}{f} = \frac{1}{s_0} + \frac{1}{s_i}\ \ \ [1/\text{mm}]
$$

$$
\frac{1}{f} = (n-1) \left(\frac{1}{R_1}-\frac{1}{R_2} + \frac{n-1}{n} \frac{CT}{R_1R_2}   \right) \ \ \ [1/\text{mm}]\\
\\
\phi = \phi_1 + \phi_2 -\phi_1 \phi_2 L = \frac{1}{\text{EFL}}  \ \ \ [1/\text{m}]
$$

$$
\text{NA} = \frac{1}{2(f/\#)}\approx \frac{D}{2f} \ \ \ [\text{rad}]
$$

$$
\text{FOV} = 2\alpha = 2\tan^{-1}\left(\frac{d}{2f}\right)\approx\frac{d}{f}\ \ \ [\text{rad}]
$$



### Aberrations 

| Type            | f-number coeff.                          | Coefficient                                     | Comments                       |
| --------------- | ---------------------------------------- | ----------------------------------------------- | ------------------------------ |
| Spherical       | $\beta = \frac{K}{(f/\#)^3}$             | $B_s = K\frac{D^3}{f^2}$                        | $ K = f(n_{\text{lens}}^{-1})$ |
| Coma            | $\beta = \frac{\theta}{16(n+2)(f/\#)^2}$ | $B_c = \beta_c f = \frac{\theta D^2}{16(n+2)f}$ |                                |
| Astigmatism     | $\beta = \frac{\theta^2}{2(f/\#)}$       | $B_a = \frac{\theta^2 D}{2}$                    |                                |
| Field curvature | $\Delta z = \frac{y²}{2nf} $             | $\approx \frac{\theta^2f}{2n} [\text{mm}]$      |                                |



### Wavefront expansion

Series expansion for rotational symmetry
$$
[\ H^2\ , \ \rho^2\ ,\ H\cos\theta\ ]\\
W_{IJK}\  \Rightarrow\ H^I\cdot\rho^J\cdot\cos^K\theta
$$


| Wavefront | Order | Expansion term.                 | Description                  |
| --------- | ----- | ------------------------------- | ---------------------------- |
| W =       | 1st   | $W_{020}\rho^2$                 | Defocus                      |
|           | 1st   | +$W_{111}H\rho\cos\theta$       | Wavefront tilt               |
|           | 3rd   | +$W_{040}\rho^4$                | SA: Spherical aberration     |
|           | 3rd   | +$W_{131}H\rho^3\cos\theta$     | Coma                         |
|           | 3rd   | +$W_{222}H^2\rho^2\cos^2\theta$ | Astigmatism                  |
|           | 3rd   | +$W_{220}H^2\rho^2$             | Field curvature              |
|           | 3rd   | +$W_{311}H^3\rho\cos\theta$     | Distortion                   |
|           | 5th   | +$W_{060}\rho^6$                | 5th Order SA                 |
|           | 5th   | +$W_{151}H\rho^5\cos\theta$     | 5th Order linear Coma        |
|           | 5th   | +$W_{422}H^4\rho^2\cos^2\theta$ | 5th Order astigmatism        |
|           | 5th   | +$W_{420}H^4\rho^2$             | 5th Order field curvature    |
|           | 5th   | +$W_{511}H^5\rho\cos\theta$     | 5th Order distortion         |
|           | 5th   | +$W_{240}H^2\rho^4$             | Sagittal oblique SA          |
|           | 5th   | +$W_{242}H^2\rho^4\cos^2\theta$ | Tangential oblique SA        |
|           | 5th   | +$W_{331}H^3\rho^3\cos\theta$   | Cubic coma (Elliptical coma) |
|           | 5th   | +$W_{333}H^3\rho^3\cos^3\theta$ | Line coma (Elliptical coma)  |
|           |       | + Higher order terms            |                              |

