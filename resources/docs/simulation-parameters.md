# Simulation parameters

The simulation parameters are the rules of a world, for example how matter moves, how much energy cells need and how strong attacks are. They are stored together with every simulation. This chapter explains the parameter window, layers and radiation sources, and lists every parameter with its default value.

## The parameter window

Open the window with **Windows > Simulation parameters** (Alt+4).

- The **Overview** at the top lists the base parameters, all layers and all radiation sources. Select an entry to show its parameters below.
- The tool bar loads, saves, copies and pastes parameter sets, adds layers and radiation sources, and clones, deletes and reorders them. **Open in new window** shows the selected entry in a separate window. This also works for the base parameters.
- The **filter field** shows only the parameters whose names contain the given text, regardless of upper and lower case. If the name of a group contains the text, the whole group is shown. A parameter such as **Shape** also stays visible if one of its sub-parameters matches, for example **Core radius**.
- Every changed parameter has a button to revert it to its **reference value**. The reference values are the parameters at the time the simulation was loaded or created. **Open parameters** and **Paste parameters** replace them with the new parameter set. Resizing the world and **Mass operations** without **Restrict to selection** also make the current parameters the new reference values.
- **Paste reference parameters** replaces only the reference values with the copied parameter set, which is useful for comparing two parameter sets. Both sets must have the same number of layers and radiation sources.
- **Expert settings** at the bottom of the window has a checkbox for the optional **Object color transition rules** group, which is off by default. When the checkbox is on, the group appears among the parameters as **Expert settings: Object color transition rules**.
- Hover over the question mark after a parameter name to read its tooltip.

Parameter sets can be saved as *.settings.json* files and loaded into other simulations.

## Values per color

Many parameters can be set separately for each of the ten customization colors. For example, green cells can live long while red cells attack strongly. Such parameters first show a single control for all colors. The plus button in front of it expands the control into one row per color, and the minus button collapses it again. As long as the control is collapsed, a change sets all colors to the same value. If the colors have different values, the collapsed control shows this. A slider, for example, then shows the range of the values. The food chain color matrix and the customization transition matrix are tables with one value for each pair of colors.

## Layers and radiation sources

The **base parameters** apply to the whole world. **Layers** override some of them in a part of the world:

- A layer has a **position**, an optional **velocity** with which it moves through the world, and a **shape**: a circle with a core radius or a rectangle with a core size. Around the core, the effect fades out over the **fade-out radius**.
- The **opacity** blends the layer with what lies below it. At 1, the layer completely replaces the values in its core. At 0, it has no effect.
- Within a layer, a parameter is only overridden if its override is switched on. All other parameters keep the value of the base or of other layers.
- If layers overlap, the later layer in the list takes precedence for the parameters it overrides. Use the arrow buttons to change the order. Force fields follow the same rule, so overlapping force fields do not add up. The color transition rules work differently. See [Object color transition rules](#object-color-transition-rules).
- A layer can exert a **force field** on all objects and energy particles inside it. See [Force field](#force-field).
- A layer can tint the background with its own color and draw its force field. This is often used for [generative art](generative-art.md).

The following parameters can be overridden by layers: background color, friction, rigidity of solids, maximum force, fusion velocity, absorption factor, radiation type I strength, minimum energy, decay rate of dying cells, attacker energy cost, food chain color matrix, size protection and color transition rules. In addition, **Disable radiation sources** switches off the radiation sources whose center lies in the core of the layer. This only works at an opacity of 1.

**Radiation sources** are places where energy particles appear. They have a position, a velocity, a circular or rectangular shape, a relative strength and an optional radiation angle. With hardware ray tracing, they also shine with their own brightness, color and range. See [Energy](energy.md#radiation-sources).

> **Note:** There is no gravity parameter. Gravity, wind and currents are created by layers with a linear force field. A layer that covers the whole world with a weak downward force acts as gravity.

## General

| Parameter | Default | Meaning |
| --- | --- | --- |
| Project name | | Name of the project, stored with the simulation. |
| Layer name, Source name | | Name of a layer or radiation source. |
| Opacity | 1 | For layers: how strongly the layer overrides the parameters below it. |
| Relative strength | | For radiation sources: the fraction of the released energy that is emitted at this source. The relative strengths of all sources and of the base add up to 1. |

## Visualization

| Parameter | Default | Meaning |
| --- | --- | --- |
| Background color | dark blue | Color of empty space. Layers tint their area if their override is enabled. A new layer has it enabled with a dark color of its own. |
| Customization colors | palette | The ten colors used for rendering and in all color selections. |
| Object coloring | Customization | How objects are colored: *Energy* (gray), *Customization* (the customization color), *Lineage + Customization* (the customization color with its hue shifted slightly per lineage), *Lineage* (one color per lineage) or *Creature* (one color per creature). The last three options only affect cells of creatures. Other objects keep their customization color. In all options, the brightness follows the energy. Per color: each object uses the setting of its customization color. |
| Glow | 0.3 | Strength of the bloom effect. |
| Hardware ray tracing | on | Renders the simulation with hardware ray tracing if the GPU supports it. If off, the simulation is rasterized, which needs less GPU time. |
| Fluid illumination | 0.25 | With hardware ray tracing: how strongly the cells and the radiation sources light the surrounding fluids. At 0, the fluids are not lit. At 1, they are lit most strongly. |
| Borderless rendering | off | Repeats the world like tiles in the view instead of showing black space around it. |
| Grid lines | off | Draws a grid that adapts to the zoom level. |
| Mark reference domain | on | Draws a frame around the world. |
| Show radiation center | on | For radiation sources: highlights their area. |
| Light color | white | For radiation sources: the color of their light and of their highlighted area. |
| Light brightness | 0.25 | For radiation sources with hardware ray tracing: the brightness of the light close to the area of the source. The light forms columns of light and shadow in a faint haze. At 0, the source does not shine. |
| Light range | 150 | For radiation sources: the distance from their area up to which their light reaches. |
| Shadow strength | 1 | For radiation sources with hardware ray tracing: the share of their light that objects and fluids on the way can block. At 1, objects and fluids cast full shadows. At 0, the light passes through them completely and they cast no shadows. |
| Show force field | on | For layers: darkens the background according to the force field. |

## Guided energy supply

| Parameter | Default | Meaning |
| --- | --- | --- |
| External energy amount | 0 | Energy in the external energy pool. |
| Inflow for constructors | 50 | Maximum energy per construction attempt that a constructor lacking energy receives from the pool. Per color. |
| Inflow threshold factor | 0 | Fraction of the required energy a constructor must have by itself before it may request external energy. Per color. |
| Inflow only for first offspring | off | If enabled, constructors that build new creatures receive external energy only for their first offspring. In each creature, only the first of these constructors that requests external energy receives it. Constructors that build parts of their own creature are not restricted. Per color. |
| Inflow for sources | 100 | Energy per time step and color for the radiation sources. Each radiation source receives this amount multiplied by its relative strength and emits it as energy particles of that color. The sources together therefore receive less than this amount as long as the relative strength of the base is above 0. Sources switched off by a layer receive nothing. Per color. |
| Backflow | 0 | Fraction of released energy that flows back into the pool. Per color. |
| Backflow limit | infinity | Backflow stops while the pool holds more energy than this limit. |

See [Energy](energy.md#the-external-energy-pool) for how these parameters work together.

## Location and shape

| Parameter | Default | Meaning |
| --- | --- | --- |
| Position (x,y) | | Center of a layer or radiation source. |
| Velocity (x,y) | 0, 0 | Velocity with which the layer or radiation source moves. In each time step, it moves by this velocity multiplied by the **Time step size**. |
| Shape | Circular | Circular or rectangular. |
| Core radius, Core size (width,height) | one sixth of the shorter world side | For layers: the area of full effect. For a rectangular core, width and height both start at this value. |
| Fade-out radius | one tenth of the shorter world side | For layers: the width of the border in which the effect fades out. |
| Radius, Size (width, height) | 1 or 30 x 60 | For radiation sources: the area in which particles are created. |

## Force field

These parameters exist for layers only.

| Parameter | Default | Meaning |
| --- | --- | --- |
| Field type | off, None | The checkbox in front of **Field type** switches the force field of the layer on. The types are *None*, *Radial* (objects circle around the center), *Central* (objects are drawn to the center), *Linear* (a constant force in one direction) and *Perlin noise* (a turbulent flow that changes over time). |
| Orientation | Clockwise | For radial fields: the direction of rotation. |
| Strength | 0.001 | For radial fields: strength of the rotation. Other field types have their own strength. |
| Drift angle | 0 | For radial fields: turns the force away from the tangent, so that objects are also pulled inwards or pushed outwards. |
| Strength | 0.05 | For central fields: strength of the attraction, which is strongest at a short distance from the center. |
| Angle | 0 | For linear fields: direction of the force. 0 points up, 90 right, 180 down. |
| Strength | 0.0001 | For linear fields: strength of the force. |
| Strength | 0.001 | For Perlin noise: strength of the flow. |
| Spatial structure size | 100 | For Perlin noise: size of the vortices. |
| Temporal structure size | 10000 | For Perlin noise: number of time steps after which the field is completely renewed. |

## Physics: Motion

| Parameter | Default | Meaning |
| --- | --- | --- |
| Time step size | 1 | Duration of one time step. The default of 1 is also the maximum. Smaller values make the simulation more accurate, but each time step then covers less time. |
| Smoothing length | 0.8 | Determines the range within which neighboring objects influence each other's density, pressure and viscosity. Fluids interact up to four times this distance, all other objects up to twice this distance. |
| Pressure | 0.1 | Strength of the pressure that pushes neighboring objects apart. |
| Viscosity | 0.1 | Strength of the viscosity. Larger values make motion smoother. |
| Friction | 0.001 | Fraction of the velocity lost per time step. Can be overridden by layers. |
| Inner friction | 0.6 | Slows down the motion of connected objects towards or away from each other. It is hidden in the parameter window and only appears when you filter for its name, for example with *friction*. |
| Rigidity of solids | 0 | Makes connected solids move more like rigid bodies. Can be overridden by layers. |

## Physics: Thresholds

| Parameter | Default | Meaning |
| --- | --- | --- |
| Maximum velocity | 2 | The highest velocity an object can reach. |
| Maximum force | 0.8 | Force above which an object loses its connections with a certain probability. Per color, can be overridden by layers. |
| Minimum distance | 0.3 | Objects that come closer than this are pushed apart. |
| Maximum distance | 3.6 | Connections longer than this break. Per color. For a connection between objects of two colors, the smaller of the two values applies. |
| Fusion velocity | 0.1 | Minimum relative velocity at which colliding objects connect, if at least one is sticky and both have free connection slots. Can be overridden by layers. |

## Radiation

| Parameter | Default | Meaning |
| --- | --- | --- |
| Relative strength | 1 | For the base: the fraction of released energy that is emitted next to the cell instead of at a radiation source. |
| Disable radiation sources | off | For layers: radiation sources whose center lies in the core of the layer emit no energy particles. This only works if the opacity of the layer is 1. With hardware ray tracing, the light of such sources keeps shining. |
| Absorption factor | 1 | Fraction of the energy of an incoming particle that a cell or free cell absorbs. Per color, can be overridden by layers. |
| Radiation type I: Strength | 0 | Fraction of its energy that a cell or free cell emits per time step once it is older than the minimum age. Per color, can be overridden by layers. |
| Radiation type I: Minimum age | 0 | Age from which radiation type I applies. Per color. |
| Radiation type II: Strength | 0 | Fraction of its energy that a cell or free cell emits per time step while its energy is above the threshold. Per color. |
| Radiation type II: Threshold | 500 | Energy above which radiation type II applies. Usable energy and raw energy are each compared with the threshold. Per color. |
| Minimum split energy | 50 | Energy particles with more energy than this split into two after a while. Per color. |
| Radiation angle | off | For radiation sources: if enabled, all particles fly in this direction. 0 points up. |

## Cell life cycle

| Parameter | Default | Meaning |
| --- | --- | --- |
| Maximum age | infinity | Cells older than this start dying. Per color. |
| Maximum free cell age | infinity | Free cells older than this disintegrate into energy particles. Per color. |
| Minimum energy | 50 | Cells whose usable energy falls below this value start dying. Free cells whose energy falls below it disintegrate with the decay rate of dying cells. Per color, can be overridden by layers. |
| Normal energy | 100 | The reference energy of a cell: the cost of a new cell, the threshold for sharing energy and for depots, and the energy at which particles may turn into free cells. Per color. |
| Decay rate of dying cells | 0.001 | Probability per time step that a dying cell disintegrates. Free cells with less than the minimum energy disintegrate with the same probability. Per color, can be overridden by layers. |
| Energy to cell free transformation | off | Energy particles turn into free cells when they reach the normal energy. |

## Cell construction

| Parameter | Default | Meaning |
| --- | --- | --- |
| Connection distance | 3.5 | When a constructor places the first cell of a construction or of a further branch, the construction is postponed if the new connection would cross existing connections within this distance. Later cells are not checked. Per color. |

## Mutations

| Parameter | Default | Meaning |
| --- | --- | --- |
| New lineage threshold | 0.25 | When the mutations that a creature has accumulated within its current lineage exceed this value, it founds a new lineage. The inspection window shows them as **Mutations (in lineage)**. |

## Meta-mutations

Meta-mutations change the mutation rates stored in genomes whose **Apply meta-mutations** option is enabled. Each sigma parameter is the standard deviation of the random change of the [main probability](evolution.md#ingredient-2-variation) of one kind of mutation. Like all mutations, this change happens once per creature, shortly before it builds its first offspring, so all offspring of a creature inherit the same rates. All sigma parameters are 0 by default, which means the mutation rates stay fixed.

| Group | Parameters |
| --- | --- |
| Meta-mutations: Properties | Neuron sigma, Connection sigma, Cell type property sigma, Constructor sigma, Geometry sigma |
| Meta-mutations: Cell identity | Cell type sigma, Cell type mode sigma, Void sigma, Customization sigma, Customization transition matrix |
| Meta-mutations: Gene structure | Add node sigma, Delete node sigma, Extend gene sigma, Trim gene sigma, Copy node section sigma, Move node section sigma |
| Meta-mutations: Genome structure | Add gene sigma, Duplicate gene sigma, Delete gene sigma, Swap gene sigma |

The **Customization transition matrix** is not a sigma parameter. It decides which color may replace which other color in a customization mutation. Rows are the old colors, columns the new ones. By default, every color may be replaced by any other color. The matrix applies to every customization mutation, even in genomes in which **Apply meta-mutations** is switched off.

## Cell type: Attacker

| Parameter | Default | Meaning |
| --- | --- | --- |
| Energy cost | 0 | Energy an attacker emits per attack attempt. Per color, can be overridden by layers. |
| Food chain color matrix | 1 | How much energy a cell of one color (row) can gain from a cell of another color (column). 0 means that cells of the row color cannot digest cells of the column color at all. Can be overridden by layers. |
| Attack strength | 0.05 | Fraction of the energy of a target that is stolen per time step. |
| Same lineage protection | 0 | Reduction of the stolen energy if the target belongs to the same lineage. Per color. |
| Size protection | 0 | Reduction of the stolen energy if the target creature has more cells than the attacking creature. Can be overridden by layers. |
| Attack radius | 2 | Maximum distance of an attack. Per color. |

## Cell type: Digestor

| Parameter | Default | Meaning |
| --- | --- | --- |
| Max raw energy conductivity | 3 | Scales the limit for the raw energy that a digestor takes in from a connected cell per time step. The limit is three times this value multiplied by the **Energy conductivity** property of the digestor. If the sending cell is also a digestor, the limits of both digestors add up. Per color. |
| Max raw energy conversion | 0.1 | Upper limit for the raw energy a digestor converts into usable energy per time step. Per color. |

## Cell type: Defender

| Parameter | Default | Meaning |
| --- | --- | --- |
| Anti-attacker strength | 0.5 | The stolen energy is divided by (1 + this value) for each defender in *Anti-attacker* mode that is the target or is connected to it. Per color, where the color of the attacking cell selects the value. |
| Anti-injector strength | 0.5 | Each defender in *Anti-injector* mode that is connected to the target raises the injection cost by this fraction of the injector's **Energy cost**. With the default cost of 250, one defender raises it to 375. Per color, where the color of the injecting cell selects the value. |

## Cell type: Injector

| Parameter | Default | Meaning |
| --- | --- | --- |
| Energy cost | 250 | Energy an injector emits per successful injection. Per color. |
| Injection radius | 3 | Maximum distance of an injection. Per color. |

## Cell type: Muscle

| Parameter | Default | Meaning |
| --- | --- | --- |
| Energy cost | 0 | Energy a muscle emits when it acts, scaled by its activation. A muscle that acts with a negative activation emits nothing. Per color. |
| Movement acceleration | 1 | Strength of muscles in *Direct movement* mode. Per color. |
| Crawling acceleration | 1 | Strength of muscles in the crawling modes. Per color. |
| Bending acceleration | 1 | Strength of muscles in *Auto bending* and *Manual bending* modes. It has no effect in *Angle bending* mode. Per color. |

## Cell type: Sensor

| Parameter | Default | Meaning |
| --- | --- | --- |
| Radius | 255 | Upper limit for the scan range of sensors. Per color. |

## Cell type: Reconnector

| Parameter | Default | Meaning |
| --- | --- | --- |
| Radius | 2 | Maximum distance at which reconnectors connect. Per color. |

## Cell type: Detonator

| Parameter | Default | Meaning |
| --- | --- | --- |
| Blast radius | 10 | Radius of the explosion. The shock wave reaches eight times as far. Per color. |
| Chain explosion probability | 0.2 | Probability that an explosion triggers other detonators in the blast radius. Per color. |

## Object color transition rules

This expert setting is off by default. When it is on, the **Target color and duration** parameter gives every color a target color and a duration. All durations are infinite by default, so no color changes until a duration is set. A cell or free cell whose age exceeds the duration of its color switches to the target color, and its age is reset to 0. This also restarts the count towards **Maximum age**, **Maximum free cell age** and **Radiation type I: Minimum age**. Chains and cycles of colors are possible.

Layers show their own rules, but unlike other layer parameters they have no checkbox to switch the override on. Rules entered for a layer therefore only take effect if an [AI agent](ai-agents.md) or a *.settings.json* file switches their override on. They then apply only in the core of a layer with an opacity of 1. Where such layers overlap, the first layer in the list applies, not the last one as for other parameters.
