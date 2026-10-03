# Glossary

**Activation function.** The function that a neuron applies to the weighted sum of its inputs, for example *Tanh* or *Identity*. See [Neural networks and signals](neural-networks.md#activation-functions).

**Auto trigger.** A setting of sensors and constructors that makes them act on their own, without a signal in channel #0.

**Backflow.** The fraction of released energy that flows back into the external energy pool. See [Energy](energy.md#the-external-energy-pool).

**Base parameters.** The simulation parameters that apply to the whole world, unless a layer overrides them.

**Branch.** One of several copies of a gene that a constructor builds around itself.

**Cell.** An object that is alive: it has energy, a cell type, a neural network and belongs to a creature with a genome. See [How life works](how-life-works.md).

**Cell type.** The special ability of a cell, for example sensor, muscle or attacker. See [Cell types](cell-types.md).

**Channel.** One of the eight values #0 to #7 of a signal.

**Concatenation.** One of several copies of a gene that a constructor builds in a row.

**Connection.** An elastic bond between two objects. Cells exchange energy and signals over connections. An object can have up to six connections.

**Connection weight.** How strongly a cell listens to the signal of a connected cell.

**Constructor.** An addition to a cell that builds new cells from a gene. See [Genomes and construction](genomes.md#how-construction-works).

**Creature.** A network of cells built from the same genome.

**Customization color.** One of ten colors that an object can have. Many simulation parameters can be set per color.

**Cycle.** The rhythm in which cells act: every 6 time steps each cell runs its neural network and its cell type.

**Edit mode.** The mode in which the mouse selects, creates and changes objects. Switched with Alt+E.

**Energy particle.** Free energy that drifts through the world and is absorbed by cells.

**External energy pool.** An energy store outside the world that can supply constructors and radiation sources. See [Energy](energy.md#the-external-energy-pool).

**Flashback.** A snapshot of the world in memory that can be restored, created in the window Temporal control.

**Fluid.** Particles without connections that flow like a liquid.

**Free cell.** Organic matter without a genome. Food for creatures.

**Front direction.** The direction that a creature regards as its front. Sensors, muscles and communicators measure directions relative to it.

**Gene.** A part of a genome that describes a group of cells and its shape.

**Genome.** The blueprint of a creature, consisting of genes. See [Genomes and construction](genomes.md).

**Head cell.** The first cell of a creature. It defines the front direction and keeps the creature together.

**Layer.** A region of the world in which some simulation parameters are overridden and a force field can act. See [Simulation parameters](simulation-parameters.md#layers-and-radiation-sources).

**Lineage.** A group of related creatures. A creature founds a new lineage when its accumulated mutations exceed a threshold.

**MCP.** The Model Context Protocol, through which AI agents control ALIEN. See [AI agents](ai-agents.md).

**Memory neuron.** One of four outputs of a neural network that keep their value from cycle to cycle and are only visible to the cell itself.

**Meta-mutation.** A random change of the mutation rates stored in a genome.

**Mutation.** A random change of a genome when it is passed on to an offspring. See [Evolution experiments](evolution.md#ingredient-2-variation).

**Node.** The description of a single cell in a gene.

**Normal energy.** The reference energy of a cell, 100 by default. Building a cell costs this amount.

**Object.** Any particle of matter: a solid, a fluid, a free cell or a cell.

**Offspring.** A creature built by another creature.

**Radiation source.** A place where energy particles appear. See [Energy](energy.md#radiation-sources).

**Raw energy.** Unprocessed energy in a cell from absorbed particles or attacks. Digestors convert it into usable energy.

**Reference value.** The value a simulation parameter had when the simulation was loaded. Changed parameters can be reverted to it.

**Root gene.** The first gene of a genome, which describes the body of the creature.

**Seed.** A single cell with a constructor that builds a creature from a genome.

**Separation.** A constructor setting that lets the finished construction detach and become a new creature.

**Shape generator.** The setting of a gene that arranges its nodes into a pattern, for example a segment or a hexagon.

**Signal.** The eight values that a cell outputs in each cycle and that its neighbors read in the next cycle.

**Solid.** Particles connected to elastic bodies such as rocks and walls.

**Static.** A property of objects that keeps them in place and protects them from forces and aging.

**Sticky.** A property of objects that lets them connect to other objects on contact.

**Telemetry.** Inputs of a neural network that describe the cell itself: energy, attacks, age and velocity.

**Time step.** The smallest unit of time in the simulation.

**Usable energy.** The energy that keeps a cell alive and pays for its actions.

**Void.** A placeholder cell type that disappears after construction.
