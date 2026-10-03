# Simulation parameters

The simulation parameters are the laws of nature of a world: how matter moves, how much energy cells need, how strong attacks are and much more. They are stored together with every simulation. This chapter explains the parameter window, layers and radiation sources, and lists every parameter with its default value.

## The parameter window

Open the window with **Windows > Simulation parameters** (Alt+4).

- The **overview** at the top lists the base parameters, all layers and all radiation sources. Select an entry to show its parameters below.
- The tool bar loads, saves, copies and pastes parameter sets, adds layers and radiation sources, and clones, deletes and reorders them. **Open in new window** shows a layer or source in a separate window.
- The **filter field** shows only parameters whose names contain the given text.
- Every changed parameter has a button to revert it to its **reference value**, which is the value it had when the simulation was loaded. **Paste reference parameters** sets new reference values from the clipboard, which is useful for comparing two parameter sets.
- **Expert settings** contains groups that are switched off by default, such as the color transition rules.
- Hover over a parameter to read its tooltip.

Parameter sets can be saved as *.settings.json* files and loaded into other simulations.

## Values per color

Many parameters can be set separately for each of the ten customization colors. Such parameters show a row of values, one per color. This allows worlds in which, for example, green cells live long and red cells attack strongly. For the food chain color matrix and the customization transition matrix, a whole table of colors against colors is set.

## Layers and radiation sources

The **base parameters** apply to the whole world. **Layers** override some of them in a part of the world:

- A layer has a **position**, an optional **velocity** with which it moves through the world, and a **shape**: a circle with a core radius or a rectangle with a core size. Around the core, the effect fades out over the **fade-out radius**.
- The **opacity** blends the layer with what lies below it. At 1 the layer replaces the values in its core completely, at 0 it has no effect.
- Within a layer, a parameter is only overridden if its override is switched on. All other parameters keep the base value.
- If layers overlap, the later layer in the list takes precedence. Use the arrow buttons to change the order.
- A layer can exert a **force field** on all objects and energy particles inside it, see the group *Force field*.
- A layer can tint the background with its own color and draw its force field, which is often used for art.

The following parameters can be overridden by layers: background color, friction, rigidity of solids, maximum force, fusion velocity, absorption factor, radiation type I strength, minimum energy, decay rate of dying cells, the attacker energy cost, food chain color matrix and size protection, the color transition rules and **Disable radiation sources**, which switches off all radiation sources inside the layer.

**Radiation sources** are places where energy particles appear. They have a position, a velocity, a circular or rectangular shape, a relative strength and an optional radiation angle. See [Energy](energy.md#radiation-sources).

> **Note:** There is no gravity parameter. Gravity, wind and currents are created by layers with a linear force field. A layer that covers the whole world with a weak downward force acts as gravity.

## General

| Parameter | Default | Meaning |
| --- | --- | --- |
| Project name | | Name of the project, stored with the simulation. |
| Layer name, Source name | | Name of a layer or radiation source. |
| Opacity | 1 | For layers: how strongly the layer overrides the parameters below it. |
| Relative strength | | For radiation sources: the fraction of the released energy that is emitted at this source. All relative strengths and the base add up to 1. |

## Visualization

| Parameter | Default | Meaning |
| --- | --- | --- |
| Background color | dark blue | Color of empty space. Layers tint their area if their override is enabled. |
| Customization colors | palette | The ten colors used for rendering and in all color selections. |
| Object coloring | Customization | How objects are colored: by energy, by customization color, by lineage or by creature. |
| Glow | 0.3 | Strength of the bloom effect. |
| Borderless rendering | off | Repeats the world periodically in the view. |
| Grid lines | off | Draws a grid that adapts to the zoom level. |
| Mark reference domain | on | Draws a frame around the world. |
| Show radiation center | on | For radiation sources: highlights their area. |
| Show force field | on | For layers: darkens the background according to the force field. |

## Guided energy supply

| Parameter | Default | Meaning |
| --- | --- | --- |
| External energy amount | 0 | Energy in the external pool. |
| Inflow for constructors | 50 | Maximum energy per construction attempt that a constructor lacking energy receives from the pool. |
| Inflow threshold factor | 0 | Fraction of the required energy a constructor must have by itself before it may request external energy. |
| Inflow only for first offspring | off | Only constructors without offspring receive external energy. |
| Inflow for sources | 100 | Energy per time step that flows from the pool to the radiation sources. |
| Backflow | 0 | Fraction of released energy that flows back into the pool. |
| Backflow limit | infinity | Backflow stops while the pool holds more energy. |

See [Energy](energy.md#the-external-energy-pool) for how these parameters work together.

## Location and shape

| Parameter | Default | Meaning |
| --- | --- | --- |
| Position (x,y) | | Center of a layer or radiation source. |
| Velocity (x,y) | 0, 0 | Movement per time step. |
| Shape | Circular | Circular or rectangular. |
| Core radius, Core size | 100 | For layers: the area of full effect. |
| Fade-out radius | 100 | For layers: the width of the border in which the effect fades out. |
| Radius, Size | 1 or 30 x 60 | For radiation sources: the area in which particles are created. |

## Force field

These parameters exist for layers only.

| Parameter | Default | Meaning |
| --- | --- | --- |
| Field type | None | *None*, *Radial* (objects circle around the center), *Central* (objects are drawn to the center), *Linear* (a constant force in one direction) or *Perlin noise* (a turbulent flow that changes over time). |
| Orientation | Clockwise | For radial fields: the direction of rotation. |
| Strength | 0.001 | For radial fields: strength of the rotation. Other field types have their own strength. |
| Drift angle | 0 | For radial fields: turns the force away from the tangent, so that objects are also pulled inwards or pushed outwards. |
| Strength | 0.05 | For central fields: strength of the attraction, strongest a short distance from the center. |
| Angle | 0 | For linear fields: direction of the force. 0 points up, 90 right, 180 down. |
| Strength | 0.0001 | For linear fields: strength of the force. |
| Strength | 0.001 | For Perlin noise: strength of the flow. |
| Spatial structure size | 100 | For Perlin noise: size of the vortices. |
| Temporal structure size | 10000 | For Perlin noise: number of time steps after which the field is completely renewed. |

## Physics: Motion

| Parameter | Default | Meaning |
| --- | --- | --- |
| Time step size | 1 | Duration of one time step. Smaller values are more accurate, larger values can become unstable. |
| Smoothing length | 0.8 | Range within which neighboring particles influence each other's density, pressure and viscosity. |
| Pressure | 0.1 | Strength of the pressure that pushes neighboring objects apart. |
| Viscosity | 0.1 | Strength of the viscosity. Larger values make motion smoother. |
| Friction | 0.001 | Fraction of the velocity lost per time step. Can be overridden by layers. |
| Rigidity of solids | 0 | Makes connected solids move more like rigid bodies. Can be overridden by layers. |

## Physics: Thresholds

| Parameter | Default | Meaning |
| --- | --- | --- |
| Maximum velocity | 2 | The highest velocity an object can reach. |
| Maximum force | 0.8 | Force above which an object loses its connections with a certain probability. Per color, can be overridden by layers. |
| Minimum distance | 0.3 | Objects that come closer are pushed apart. |
| Maximum distance | 3.6 | Connections longer than this break. Per color. |
| Fusion velocity | 0.1 | Minimum relative velocity at which colliding objects connect, if at least one is sticky and both have free connection slots. |

## Radiation

| Parameter | Default | Meaning |
| --- | --- | --- |
| Relative strength | 1 | For the base: the fraction of released energy that is emitted next to the cell instead of at a radiation source. |
| Disable radiation sources | off | For layers: switches off all radiation sources inside the layer. |
| Absorption factor | 1 | Fraction of the energy of an incoming particle that a cell absorbs. Per color. |
| Radiation type I: Strength | 0 | Fraction of its energy that a cell emits per time step once it is older than the minimum age. Per color. |
| Radiation type I: Minimum age | 0 | Age from which radiation type I applies. Per color. |
| Radiation type II: Strength | 0 | Fraction of its energy that a cell emits per time step while its energy is above the threshold. Per color. |
| Radiation type II: Threshold | 500 | Energy above which radiation type II applies. Usable and raw energy count separately. Per color. |
| Minimum split energy | 50 | Energy particles with more energy split into two after a while. Per color. |
| Radiation angle | off | For radiation sources: if enabled, all particles fly in this direction. 0 points up. |

## Cell life cycle

| Parameter | Default | Meaning |
| --- | --- | --- |
| Maximum age | infinity | Cells older than this start dying. Per color. |
| Maximum free cell age | infinity | Older free cells disintegrate into energy particles. Per color. |
| Minimum energy | 50 | Cells with less usable energy start dying. Free cells with less energy disintegrate. Per color, can be overridden by layers. |
| Normal energy | 100 | The reference energy of a cell: the cost of a new cell, the threshold for sharing energy and for depots, and the energy at which particles may turn into free cells. Per color. |
| Decay rate of dying cells | 0.001 | Probability per time step that a dying cell disintegrates. Per color, can be overridden by layers. |
| Energy to cell free transformation | off | Energy particles turn into free cells when they reach the normal energy. |

## Cell construction

| Parameter | Default | Meaning |
| --- | --- | --- |
| Connection distance | 3.5 | When a constructor starts a new branch, the construction is postponed if the new connection would cross existing connections within this distance. Per color. |

## Mutations

| Parameter | Default | Meaning |
| --- | --- | --- |
| New lineage threshold | 0.25 | When the accumulated mutations of a creature exceed this value, it founds a new lineage. |

## Meta-mutations

Meta-mutations change the mutation rates stored in genomes whose option **Apply meta-mutations** is enabled. Each parameter is the standard deviation of the random change of one kind of mutation probability per offspring. All are 0 by default, which means the mutation rates stay fixed.

| Group | Parameters |
| --- | --- |
| Meta-mutations: Properties | Neuron sigma, Connection sigma, Cell type property sigma, Constructor sigma, Geometry sigma |
| Meta-mutations: Cell identity | Cell type sigma, Cell type mode sigma, Void sigma, Customization sigma, Customization transition matrix |
| Meta-mutations: Gene structure | Add node sigma, Delete node sigma, Extend gene sigma, Trim gene sigma, Copy node section sigma, Move node section sigma |
| Meta-mutations: Genome structure | Add gene sigma, Duplicate gene sigma, Delete gene sigma, Swap gene sigma |

The **Customization transition matrix** decides which color may replace which other color in a customization mutation. Rows are the old colors, columns the new ones.

## Cell type: Attacker

| Parameter | Default | Meaning |
| --- | --- | --- |
| Energy cost | 0 | Energy an attacker emits per attack attempt. Per color, can be overridden by layers. |
| Food chain color matrix | 1 | How much energy a cell of one color (row) can gain from a cell of another color (column). 0 means it cannot digest it at all. Can be overridden by layers. |
| Attack strength | 0.05 | Fraction of the energy of a target that is stolen per time step. |
| Same lineage protection | 0 | Reduction of the stolen energy if the target belongs to the same lineage. Per color. |
| Size protection | 0 | Reduction of the stolen energy if the target creature has more cells than the attacker. Can be overridden by layers. |
| Attack radius | 2 | Maximum distance of an attack. Per color. |

## Cell type: Digestor

| Parameter | Default | Meaning |
| --- | --- | --- |
| Max raw energy conductivity | 3 | Upper limit for the raw energy a digestor takes in from a connected cell or passes on. Per color. |
| Max raw energy conversion | 0.1 | Upper limit for the raw energy a digestor converts into usable energy per time step. Per color. |

## Cell type: Defender

| Parameter | Default | Meaning |
| --- | --- | --- |
| Anti-attacker strength | 0.5 | The stolen energy is divided by (1 + this value) for each defender at the target. Per color. |
| Anti-injector strength | 0.5 | The injection cost increases by this fraction for each defender at the target. Per color. |

## Cell type: Injector

| Parameter | Default | Meaning |
| --- | --- | --- |
| Energy cost | 250 | Energy an injector emits per successful injection. Per color. |
| Injection radius | 3 | Maximum distance of an injection. Per color. |

## Cell type: Muscle

| Parameter | Default | Meaning |
| --- | --- | --- |
| Energy cost | 0 | Energy a muscle emits when it acts, scaled by its activation. Per color. |
| Movement acceleration | 1 | Strength of muscles in the mode *Direct movement*. Per color. |
| Crawling acceleration | 1 | Strength of muscles in the crawling modes. Per color. |
| Bending acceleration | 1 | Strength of muscles in the bending modes. Per color. |

## Cell type: Sensor, Reconnector and Detonator

| Parameter | Default | Meaning |
| --- | --- | --- |
| Sensor: Radius | 255 | Upper limit for the scan range of sensors. Per color. |
| Reconnector: Radius | 2 | Maximum distance at which reconnectors connect. Per color. |
| Detonator: Blast radius | 10 | Radius of the explosion. The shock wave reaches eight times further. Per color. |
| Detonator: Chain explosion probability | 0.2 | Probability that an explosion triggers other detonators in the blast radius. Per color. |

## Object color transition rules

This expert setting is off by default. When enabled, every color can be given a following color and a duration. Cells and free cells that have kept a color for the duration switch to the following color, and their age is reset. Chains and cycles of colors are possible. The rules can also be set per layer.
