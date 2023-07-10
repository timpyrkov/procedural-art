# procedural-art

## Examples of procedural generative art and animations in python
#

# Install requirements
```
pip install requirements.txt
```

Perlin noise lib and animations can be found at:

https://github.com/timpyrkov/pythonperlin/

https://www.instagram.com/timpyrkov/


# Cloud texture

- To genrate cloud texture we use perlin noise with large number of octaves. Then colorize it using a white-blue colormap.

- Alternatively, we can colorize it with red-dark colormap to get a Babylon-5 hyperspace-looking texture.

```
from pythonperlin import perlin
import pylab as plt

p = perlin((3,3), dens=128, octaves=4)
plt.imshow(p, cmap="Blues")
```

![](img/clouds.jpg)
![](img/starfury.jpg)

# Water caustics

- To generate water caustics we use absolute value of perlin noise (no octaves). Then contrast it with a logscale-sampled colormap.

```
logscale = np.logspace(0,-2,5)
colors = plt.cm.get_cmap("PuBuGn_r")(logscale)
cmap = LinearSegmentedColormap.from_list("caustics", colors)
```

![](img/caustics.jpg)

# Flow field

- To generate a flowfield we use Perlin noise (no octaves) as a vector field. Value at each grid node gives a direction vector (gradient). Then drop random dots and calculate how they move along the gradients. 

```
p = perlin((6,6), dens=24)
z = np.exp(2j * np.pi * p)
dx, dy = z.real, z.imag
x[k+1] = x[k] + dx
y[k+1] = y[k] + dy
```

![](img/flowfield.jpg)

# Wireframe

- To generate a smoothly morphing wireframe we use Perlin noise (no octaves) to distort each grid node. Then populate the grid with more lines along one direction (x).

```
from pythonperlin import perlin, extend
p = perlin((6,6), dens=24)
z = np.exp(2j * np.pi * p)
dx, dy = z.real, z.imag
x = extend(x + dx, 16)
y = extend(y + dy, 16)
```

![](img/wireframe.jpg)

# Marble

- To model a marble texture we use perlin noise with large number of octaves. Then contrast it using a logscale-sampled colormap.

- Marble veins can be added using a periodic function sin(x).

```
from pythonperlin import perlin, extend
p = perlin((6,6), dens=24)
z = np.exp(2j * np.pi * p)
dx, dy = z.real, z.imag
x = extend(x + dx, 16)
y = extend(y + dy, 16)
```

![](img/marble_light.jpg)
![](img/marble_dark.jpg)
![](img/marble_veins.jpg)

