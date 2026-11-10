# Glossary

**Activation function.** The function that a neuron applies to the weighted sum of its inputs plus its bias, for example *Tanh* or *Identity*. See [Neural networks and signals](neural-networks.md#activation-functions).

**Auto trigger.** A setting of sensors and constructors that makes them act on their own, without a signal in channel #0.

**Backflow.** The fraction of released energy that flows back into the external energy pool. See [Energy](energy.md#the-external-energy-pool).

**Base parameters.** The simulation parameters that apply to the whole world, unless a layer overrides them.

**Branch.** One of several copies of a gene that a constructor builds around itself.

**Cell.** An object that is alive. It has energy, a cell type and a neural network, and it belongs to a creature with a genome. See [How life works](how-life-works.md).

**Cell type.** The special ability of a cell, for example sensor, muscle or attacker. See [Cell types](cell-types.md).

**Channel.** One of the eight values of a signal, numbered #0 to #7.

**Concatenation.** One of several copies of a gene that a constructor builds in a row.

**Connection.** An elastic bond between two objects. Cells exchange energy and signals over connections. An object can have up to six connections.

**Connection weight.** How strongly a cell listens to the signal of a connected cell.

**Constructor.** An addition to a cell that builds new cells from a gene. See [Genomes and construction](genomes.md#how-construction-works).

**Creature.** A network of cells built from the same genome.

**Customization color.** One of ten colors that an object can have. Many simulation parameters can be set per color.

**Cycle.** The repeating period in which cells act. Every 6 time steps, each cell runs its neural network and its cell type.

**Edit mode.** The mode in which the mouse selects, creates and changes objects. It is switched on and off with **Alt+E**.

**Energy particle.** Free energy that drifts through the world. Cells and free cells absorb it on contact. In a cell it becomes raw energy.

**Energy reserve.** Energy that a constructor cell holds for construction, shown as **Reserved energy** in the inspection window. The constructor spends it before its usable energy. It is filled by the external energy pool and by the constructor that built the cell, if that constructor's **Provide energy** is set to *Transitive cells*.

**External energy pool.** An energy store outside the world that can supply constructors and radiation sources. See [Energy](energy.md#the-external-energy-pool).

**Flashback.** A snapshot of the world in memory. **Create flashback** in the **Temporal control** window stores it and **Load flashback** returns to it. There is only one flashback, and a new one replaces the previous one. ALIEN also creates one automatically, for example when a simulation is opened. Loading a flashback keeps the current simulation parameters, apart from the **External energy amount** and the positions of moving layers and radiation sources. See [The user interface](user-interface.md#temporal-control).

**Fluid.** Particles without connections that flow like a liquid.

**Force field.** A force that a layer exerts on the objects and energy particles inside it. The types are *Radial*, *Central*, *Linear* and *Perlin noise*. Static objects are not affected. A weak downward *Linear* field acts as gravity. See [Simulation parameters](simulation-parameters.md#force-field).

**Free cell.** Organic matter without a genome. It serves as food for creatures.

**Front direction.** The direction that a creature regards as its front. Sensors, muscles and communicators measure directions relative to it.

**Gene.** A part of a genome that describes a group of cells and their shape.

**Genome.** The blueprint of a creature, consisting of genes. See [Genomes and construction](genomes.md).

**Head cell.** The first cell of a creature. It defines the front direction and keeps the cells of the creature alive.

**Layer.** A region of the world in which some simulation parameters are overridden and a force field can act. See [Simulation parameters](simulation-parameters.md#layers-and-radiation-sources).

**Lineage.** A group of related creatures that share a lineage id. A creature founds a new lineage when the mutations it has accumulated within its current lineage exceed the **New lineage threshold** parameter, 0.25 by default.

**MCP.** The Model Context Protocol, through which AI agents control ALIEN. See [AI agents](ai-agents.md).

**Memory neuron.** One of the four memory outputs of a neural network. Its value is only visible to the cell itself and serves as an input in the next cycle. It is only recalculated when channel #0 of the incoming signals is greater than 0.1. Otherwise it keeps its value. See [Neural networks and signals](neural-networks.md#memory-outputs).

**Meta-mutation.** A random change to the mutation rates stored in a genome.

**Minimum energy.** The usable energy below which a cell starts to die, 50 by default. Free cells with less energy disintegrate.

**Mutation.** A random change to a genome during reproduction. A creature mutates its genome once, when it starts to build its first offspring. It keeps using the mutated genome itself, and all its offspring inherit it. A seed created in the genome editor passes on its genome unchanged. See [Evolution experiments](evolution.md#ingredient-2-variation).

**Mutation rates.** The probabilities and strengths of all kinds of mutation, stored in every genome. In a new genome, the main probability of every kind of mutation is zero. See [Evolution experiments](evolution.md#ingredient-2-variation).

**Navigation mode.** The mode in which the mouse only zooms and moves the view. **Alt+E** switches between navigation mode and edit mode.

**Node.** The description of a single cell in a gene.

**Normal energy.** The reference energy of a cell, 100 by default. Building a cell costs this amount.

**Object.** Any particle of matter: a solid, a fluid, a free cell or a cell.

**Offspring.** A creature built by another creature.

**Radiation.** Energy that cells and free cells emit continuously as energy particles. The **Radiation type I: Strength** and **Radiation type II: Strength** parameters control it. Both are 0 by default, so there is no radiation.

**Radiation source.** A place where energy particles appear. See [Energy](energy.md#radiation-sources).

**Raw energy.** Unprocessed energy that a cell collects from absorbed particles or attacks. Digestors convert it into usable energy.

**Reference value.** The value to which a changed simulation parameter can be reverted. Usually it is the value that the parameter had when the simulation was loaded or created. Some actions set new reference values, for example **Open parameters**, **Paste parameters** and resizing the world. See [Simulation parameters](simulation-parameters.md#the-parameter-window).

**Replicator.** A creature whose genome contains a constructor that builds the root gene with **Separation** switched on, so that it reproduces. See [Evolution experiments](evolution.md#ingredient-1-replicators).

**Root gene.** The first gene of a genome, which describes the body of the creature.

**Seed.** A single cell with a constructor that builds creatures from a genome.

**Separation.** A constructor setting that lets the finished construction detach and become a new creature. A constructor that builds the root gene creates a new creature even without separation, but this offspring stays attached to its parent.

**Shape generator.** The setting of a gene that arranges its nodes into a pattern, for example a segment or a hexagon.

**Signal.** The set of eight values that a cell outputs in each cycle and that its neighbors read in the next cycle.

**Solid.** Particles that are connected to each other and form elastic bodies such as rocks and walls.

**Static.** A property of objects that protects them from forces, aging and death. A static object keeps its velocity, so it stays in place only as long as its velocity is zero.

**Sticky.** A property of objects that lets them form a new connection when they collide with another object. It is enough if one of the two objects is sticky. A connection only forms if their relative velocity is at least the **Fusion velocity** parameter, 0.1 by default, and both objects have a free connection slot. Static objects and fluids never connect this way.

**Telemetry.** Inputs of a neural network that describe the cell itself: energy, attacks, age and velocity.

**Time step.** The smallest unit of time in the simulation.

**Trigger.** Many cell types act only when they are triggered. A cell is triggered when the absolute value of channel #0 of its signal is at least 0.1. Sensors and constructors can also trigger themselves, see **Auto trigger**.

**Usable energy.** The energy that keeps a cell alive and pays for its actions.

**Void.** A placeholder cell type that disappears after construction.
