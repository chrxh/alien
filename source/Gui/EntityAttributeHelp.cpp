#include "EntityAttributeHelp.h"

#include <array>
#include <string_view>

#include <Fonts/IconsFontAwesome5.h>

namespace
{
    struct Entry
    {
        EntityAttribute attribute;
        std::string_view description;
    };

    constexpr auto Descriptions = std::to_array<Entry>({

        // Object
        {EntityAttribute::ObjectId, "Unique 64 bit id of the object. It is assigned by the simulation and cannot be changed."},
        {EntityAttribute::Position, "Position of the object in world coordinates."},
        {EntityAttribute::Velocity, "Velocity of the object in world units per time step."},
        {EntityAttribute::Stiffness, "Resistance of the object against deformation of its connections."},
        {EntityAttribute::Color, "Customization color of the object. Many simulation parameters can be defined per color."},
        {EntityAttribute::Static, "A static object is immovable and immortal. It does not age and is not affected by forces."},
        {EntityAttribute::Sticky,
         "If enabled, the object can spontaneously form new connections on contact. It is sufficient if one of the two objects is sticky."},
        {EntityAttribute::ObjectType,
         "Type of the object."
         "\n" ICON_FA_CHEVRON_RIGHT " Solid: inorganic rigid particle. It blocks the scan rays of sensor cells."
         "\n" ICON_FA_CHEVRON_RIGHT " Fluid: inorganic freely flowing particle without connections."
         "\n" ICON_FA_CHEVRON_RIGHT " Free cell: organic substance without a genome. It can serves as food."
         "\n" ICON_FA_CHEVRON_RIGHT " Cell: cell of a creature with genome, neural network and cell type."},
        {EntityAttribute::ConnectedId, "Id of the object that is attached by this connection."},
        {EntityAttribute::ConnectionDistance, "Reference distance of the connection. A deviation of the actual distance generates restoring forces."},
        {EntityAttribute::ConnectionRefAngle, "Reference angle between this connection and the previous one. A deviation generates restoring forces."},

        // Energy particle
        {EntityAttribute::ParticleId, "Unique 64 bit id of the energy particle."},
        {EntityAttribute::ParticleEnergy, "Energy of the particle. Cells can absorb it."},

        // Solid, fluid and free cell
        {EntityAttribute::SolidEnergy, "Energy stored in the solid particle."},
        {EntityAttribute::FluidEnergy, "Energy stored in the fluid particle."},
        {EntityAttribute::FluidGlow, "Additional brightness of the fluid particle during rendering."},
        {EntityAttribute::FreeCellEnergy,
         "Energy of the free cell. Below the minimum cell energy the cell decays."},
        {EntityAttribute::FreeCellAge, "Age of the free cell in time steps."},

        // Cell
        {EntityAttribute::CellUsableEnergy,
         "Energy that keeps the cell alive and can be use for creating new ones. Below the minimum cell energy the cell starts dying."},
        {EntityAttribute::CellRawEnergy, "Unprocessed energy. Attacker cells gain raw energy while digestor cells convert it into usable energy."},
        {EntityAttribute::CellFrontAngle,
         "Angle between the first connection and the front direction of the creature. It orients muscle, sensor and communicator cells."},
        {EntityAttribute::CellAge, "Age of the cell in time steps."},
        {EntityAttribute::CellState,
         "State of the cell in its life cycle."
         "\n" ICON_FA_CHEVRON_RIGHT " Ready: The cell is in normal operating mode. Neural networks and additional cell functions can be executed."
         "\n" ICON_FA_CHEVRON_RIGHT " Under construction: The cell belongs to an unfinished construction and is inactive."
         "\n" ICON_FA_CHEVRON_RIGHT " Being activated: Transitional state after the construction. The cell becomes ready in the next step."
         "\n" ICON_FA_CHEVRON_RIGHT " Dying: The cell decays with the probability given by the simulation parameter 'Cell death probability'."
         "\n" ICON_FA_CHEVRON_RIGHT " Instant dying: The cell is removed in the next time step."},
        {EntityAttribute::CellNodeIndex, "Index of the genome node from which this cell was built."},
        {EntityAttribute::CellParentNodeIndex, "Node index of the cell that built this cell."},
        {EntityAttribute::CellGeneIndex, "Index of the gene from which this cell was built."},
        {EntityAttribute::CellConcatenationIndex, "Index of the concatenation in which this cell was built."},
        {EntityAttribute::CellBranchIndex, "Index of the branch in which this cell was built."},
        {EntityAttribute::CellActivationTime,
         "Remaining time steps until the cell starts executing its function. As long as the value is greater than 0, the cell is inactive."},
        {EntityAttribute::CellHeadCell,
         "Marks the head cell of the creature. Cells that lose connection to head cells die off."},
        {EntityAttribute::CellType,
         "Function that the cell performs after its neural network has been executed."
         "\n" ICON_FA_CHEVRON_RIGHT " Base: No special function. Only the neural network is evaluated."
         "\n" ICON_FA_CHEVRON_RIGHT " Depot: Stores usable energy and releases it again on demand. Both storing and releasing are controlled by neural activities."
         "\n" ICON_FA_CHEVRON_RIGHT " Sensor: Scans the environment and reports direction, distance and mass of a match."
         "\n" ICON_FA_CHEVRON_RIGHT " Generator: Generates a periodic signal."
         "\n" ICON_FA_CHEVRON_RIGHT " Attacker: Steals energy from free cells or from cells of other creatures."
         "\n" ICON_FA_CHEVRON_RIGHT " Injector: Injects its own genome into a foreign cell that has a constructor."
         "\n" ICON_FA_CHEVRON_RIGHT " Muscle: Generates movement by bending connections or changing their length."
         "\n" ICON_FA_CHEVRON_RIGHT " Defender: Weakens nearby attacker or injector cells from other creatures."
         "\n" ICON_FA_CHEVRON_RIGHT " Reconnector: Establishes or removes connections to other objects."
         "\n" ICON_FA_CHEVRON_RIGHT " Detonator: Explodes after a countdown and pushes away surrounding objects."
         "\n" ICON_FA_CHEVRON_RIGHT " Digestor: Converts raw energy into usable energy and conducts it to other digestor cells."
         "\n" ICON_FA_CHEVRON_RIGHT " Memory: Delays, records or smooths the signal."
         "\n" ICON_FA_CHEVRON_RIGHT " Communicator: Transmits the signal to communicator cells of other creatures."
         "\n" ICON_FA_CHEVRON_RIGHT " Void: Dissolves immediately after the cell network is finished and passes its energy to the neighboring cell. Serves as temporary scaffolding for constructing complex shapes."},
        {EntityAttribute::CellHasConstructor, "If enabled, the cell can build other cells specified by a gene."},

        // Genome
        {EntityAttribute::GenomeName, "Name of the genome. Every creature that is built from it inherits the name."},
        {EntityAttribute::GenomeFrontAngle,
         "Angle between the first connection of the head cell and the front direction of the creature. From the head cell the front direction is propagated "
         "to all other cells."},
        {EntityAttribute::GenomeResistanceToInjection, "If enabled, injector cells cannot overwrite the genome of creatures built from it."},
        {EntityAttribute::GenomeApplyMetaMutations,
         "If enabled, the mutation rates of this genome are themselves mutated. The step size comes from the simulation parameters under 'Meta-mutations'."},
        {EntityAttribute::GenomeMutationRatesEdit, "Opens the editor for all mutation rates. The rows below only list the rates that are currently active."},

        // Gene
        {EntityAttribute::GeneName, "Name of the gene. It is only used for the display in the structure list."},
        {EntityAttribute::GeneShape,
         "Determines the angles and the additional connections with which the nodes of this gene are assembled. The angles of the first and the last node "
         "come from the nodes themselves."},
        {EntityAttribute::GeneConnectionDistance, "Reference distance between two cells that are constructed one after another from this gene."},
        {EntityAttribute::GeneStiffness,
         "Stiffness of all cells constructed from this gene. It determines how strongly they resist a deformation of their connections."},
        {EntityAttribute::GeneHomogeneousCellType, "If enabled, every constructed cell of this gene uses the cell type and its properties of the first node."},

        // Mutation rates. All mutations are applied exactly once per offspring, namely when its genome is passed on.
        {EntityAttribute::MutationConnectionProbability,
         "Probability per node and connection that the weight of that connection in the neural network is changed."},
        {EntityAttribute::MutationConnectionValueChangeSigma,
         "Standard deviation of the Gaussian change of the connection weight. The result is limited to [-2, 2]."},
        {EntityAttribute::MutationNeuronProbability, "Probability per node and neuron that the neuron is changed."},
        {EntityAttribute::MutationNeuronWeightChangeSigma,
         "Standard deviation of the Gaussian change of every input weight of the neuron. The result is limited to [-2, 2]."},
        {EntityAttribute::MutationNeuronBiasChangeSigma, "Standard deviation of the Gaussian change of the bias of the neuron."},
        {EntityAttribute::MutationNeuronActfnChangeProbability, "Probability that the activation function of the neuron is replaced by a different one."},
        {EntityAttribute::MutationCellTypePropertiesProbability, "Probability per node that the properties of its cell type are changed."},
        {EntityAttribute::MutationCellTypePropertiesValueChangeSigma,
         "Standard deviation of the Gaussian change of the numeric properties, relative to the value range of each property."},
        {EntityAttribute::MutationCellTypePropertiesEnumChangeProbability,
         "Probability that a switch, a mode or a color selection of the cell type is changed."},
        {EntityAttribute::MutationGeometryProbability, "Probability per gene that its geometry is changed."},
        {EntityAttribute::MutationGeometryValueChangeSigma,
         "Standard deviation of the Gaussian change of the stiffness, relative to its value range. The connection distance is not mutated."},
        {EntityAttribute::MutationGeometryEnumChangeProbability, "Probability that the shape generator of the gene is replaced by a different one."},
        {EntityAttribute::MutationCellTypeModeProbability,
         "Probability per node that the mode of its cell type is switched to a different one. The mode-specific properties are reset to their default values; "
         "cell types without a mode stay unchanged."},
        {EntityAttribute::MutationCellTypeProbability,
         "Probability per node that its cell type is replaced by a different one, with all properties reset to their default values. Void nodes are not "
         "affected. With the same probability the switch 'Homogeneous cell type' of the gene is toggled."},
        {EntityAttribute::MutationCustomizationProbability,
         "Probability per genome that one used customization color is replaced everywhere by another one. Which replacements are permitted is defined by the "
         "simulation parameter 'Customization transition matrix'."},
        {EntityAttribute::MutationVoidProbability,
         "Probability per node that it is toggled between void and a random other cell type. The first and the last node of a gene are not affected."},
        {EntityAttribute::MutationExtendGeneProbability, "Probability per gene that a new node is appended at its beginning or at its end."},
        {EntityAttribute::MutationAddNodeProbability, "Probability per insertion position that a new node is inserted there."},
        {EntityAttribute::MutationTrimGeneProbability,
         "Probability per gene that its first or its last node is removed. The last remaining node of a gene is always kept."},
        {EntityAttribute::MutationDeleteNodeProbability, "Probability per node that it is removed. The last remaining node of a gene is always kept."},
        {EntityAttribute::MutationDuplicateGeneProbability,
         "Probability per gene that it is duplicated. Only genes that are referenced at least twice are duplicated; one of these references then points to the "
         "copy."},
        {EntityAttribute::MutationDeleteGeneProbability,
         "Probability per gene that it is deleted. At least one gene is kept. A reference to a deleted gene switches off the referencing constructor and "
         "points a referencing injector to the first gene."},
        {EntityAttribute::MutationCopyNodeSectionProbability,
         "Probability per gene that a contiguous section of its nodes is copied into a randomly chosen gene, which can also be the gene itself."},
        {EntityAttribute::MutationMoveNodeSectionProbability,
         "Probability per gene that a contiguous section of its nodes is moved into a randomly chosen gene. At least one node stays behind."},
        {EntityAttribute::MutationConstructorProbability, "Probability per node that the constructor properties of the node are changed."},
        {EntityAttribute::MutationConstructorValueChangeSigma,
         "Standard deviation of the Gaussian change of the numeric constructor properties, relative to the value range of each property."},
        {EntityAttribute::MutationConstructorEnumChangeProbability, "Probability that a switch or a selection of the constructor is changed."},
        {EntityAttribute::MutationConstructorToggleProbability, "Probability that the constructor of the node is switched on or off."},

        // Genome node
        {EntityAttribute::NodeAngle,
         "Angle of this cell relative to the connection to the previous cell. It is only evaluated for the first and the last node of a gene; for inner "
         "nodes the angle results from the selected shape."},
        {EntityAttribute::NodeCustomization, "Customization color of the cell. Many simulation parameters are defined per color."},
        {EntityAttribute::NodeConstructionGene, "Gene that this cell builds as a constructor. 'None' means that the cell has no constructor."},

        // Constructor
        {EntityAttribute::ConstructorAutoTriggerInterval,
         "If set, the constructor triggers itself every n time steps, with a phase that differs per creature. Without a value it has to be triggered via "
         "channel #0."},
        {EntityAttribute::ConstructorActivationTime,
         "Number of time steps for which a newly constructed cell is to remain inactive. Currently without effect: the engine does not evaluate this value."},
        {EntityAttribute::ConstructorConstructionAngle,
         "Angle of the first constructed cell relative to the connection of the constructor cell. It is only evaluated for the first cell of the first "
         "concatenation in the first branch."},
        {EntityAttribute::ConstructorProvideEnergy,
         "Determines where the energy for the construction comes from."
         "\n" ICON_FA_CHEVRON_RIGHT " Reduce cell energy: every constructed cell takes its energy from the constructor cell."
         "\n" ICON_FA_CHEVRON_RIGHT " Free: cells are built without energy cost. After the first completed offspring the setting falls back to 'Reduce cell "
         "energy'."},
        {EntityAttribute::ConstructorReservedEnergy,
         "Energy reserve of the constructor cell. It is used up for the construction before the usable energy of the cell. External energy inflow is credited "
         "here."},
        {EntityAttribute::ConstructorSeparation, "If enabled, the offspring detaches from the constructor cell after the last cell has been built."},
        {EntityAttribute::ConstructorNumBranches, "Number of cell networks that the constructor builds next to each other from the same gene."},
        {EntityAttribute::ConstructorNumConcatenations, "How often the gene is built in a row, each attached to the previous one."},
        {EntityAttribute::ConstructorGeneIndex, "Index of the gene that the constructor builds."},

        // Depot
        {EntityAttribute::DepotStorageLimit, "Maximum usable energy that the depot can store."},
        {EntityAttribute::DepotInitialStoredEnergy,
         "Energy that the depot already contains when it is constructed. The constructor has to provide this energy in addition."},
        {EntityAttribute::DepotStoredEnergy, "Usable energy that is currently stored in the depot."},

        // Sensor
        {EntityAttribute::SensorAutoTrigger,
         "If enabled, the sensor scans in every cycle. Otherwise it only scans when channel #0 exceeds the trigger threshold."},
        {EntityAttribute::SensorTagForAttackers,
         "If enabled, attacker cells of the same creature in 'Creature' mode may use the last match of this sensor as a target."},
        {EntityAttribute::SensorMode,
         "Selects what the sensor scans for."
         "\n" ICON_FA_CHEVRON_RIGHT " Detect energy: searches for accumulations of energy particles."
         "\n" ICON_FA_CHEVRON_RIGHT " Detect solid: searches for solid particles."
         "\n" ICON_FA_CHEVRON_RIGHT " Detect free cell: searches for accumulations of free cells."
         "\n" ICON_FA_CHEVRON_RIGHT " Detect creature: searches for cells of other creatures."},
        {EntityAttribute::SensorEnergyMinDensity, "Minimum density of energy particles that is required for a match. The value lies between 0 and 1."},
        {EntityAttribute::SensorFreeCellMinDensity, "Minimum density of free cells that is required for a match. The value lies between 0 and 1."},
        {EntityAttribute::SensorMinRange, "Objects that are closer than this distance are not detected."},
        {EntityAttribute::SensorMaxRange,
         "Objects that are farther away than this distance are not detected. The simulation parameter 'Sensor radius' limits the range additionally."},
        {EntityAttribute::SensorRestrictToColors, "Only objects with one of the selected customization colors are detected."},
        {EntityAttribute::SensorRestrictToLineage,
         "Restricts the detection to the relationship with the own creature."
         "\n" ICON_FA_CHEVRON_RIGHT " No: the lineage is not taken into account."
         "\n" ICON_FA_CHEVRON_RIGHT " Same lineage: only creatures of the own lineage are detected."
         "\n" ICON_FA_CHEVRON_RIGHT " Other lineage: only creatures of a foreign lineage are detected."},
        {EntityAttribute::SensorMinNumCells, "Only creatures with at least this number of cells are detected. Without a value there is no lower limit."},
        {EntityAttribute::SensorMaxNumCells, "Only creatures with at most this number of cells are detected. Without a value there is no upper limit."},

        // Generator
        {EntityAttribute::GeneratorAdditive, "If enabled, the generated value is added to channel #0 instead of overwriting it."},
        {EntityAttribute::GeneratorMinValue, "Lower level of the generated signal."},
        {EntityAttribute::GeneratorMaxValue, "Upper level of the generated signal."},
        {EntityAttribute::GeneratorTimeOffset, "Shifts the phase of the signal by this number of time steps."},
        {EntityAttribute::GeneratorMode,
         "Shape of the generated signal."
         "\n" ICON_FA_CHEVRON_RIGHT " Square signal: the maximum value is output in the first half of the period and the minimum value in the second half."
         "\n" ICON_FA_CHEVRON_RIGHT " Sawtooth signal: the output rises linearly from the minimum to the maximum value over the period."},
        {EntityAttribute::GeneratorPeriod, "Length of one cycle in time steps."},

        // Attacker
        {EntityAttribute::AttackerMode,
         "Selects the targets of the attack. The stolen energy is added to the own raw energy."
         "\n" ICON_FA_CHEVRON_RIGHT " Free cell: steals energy from free cells within the attack radius."
         "\n" ICON_FA_CHEVRON_RIGHT " Creature: steals energy from cells of other creatures. Only creatures that a sensor cell of the same creature has "
         "detected and tagged are attacked."},
        {EntityAttribute::AttackerRestrictToColors, "Only free cells with one of the selected customization colors are attacked."},

        // Injector
        {EntityAttribute::InjectorGeneIndex, "Gene index that is set in the constructor of the injected cell. It determines which gene is built there."},

        // Muscle
        {EntityAttribute::MuscleMode,
         "Selects how the muscle generates movement. Bending and crawling act on the first connection."
         "\n" ICON_FA_CHEVRON_RIGHT " Auto bending: bends back and forth on its own between the angle limits. Channel #1 sets the target direction."
         "\n" ICON_FA_CHEVRON_RIGHT " Manual bending: the bending direction follows the sign of channel #0."
         "\n" ICON_FA_CHEVRON_RIGHT " Angle bending: bends towards the direction given in channel #1 and turns the front direction along with it."
         "\n" ICON_FA_CHEVRON_RIGHT " Auto crawling: changes the connection length back and forth on its own."
         "\n" ICON_FA_CHEVRON_RIGHT " Manual crawling: the change of the connection length follows the sign of channel #0."
         "\n" ICON_FA_CHEVRON_RIGHT " Direct movement: accelerates the cell directly in the direction given in channel #1."},
        {EntityAttribute::MuscleMaxAngleDeviation, "Maximum deviation from the initial angle, as a fraction of the available angular range."},
        {EntityAttribute::MuscleForwardBackwardRatio,
         "Distributes the speed between the forward and the backward stroke. Since the faster stroke generates more thrust, this determines the direction of "
         "movement. At 0.5 both strokes cancel each other out."},
        {EntityAttribute::MuscleAttractionRepulsionRatio, "Distributes the speed between the attracting and the repelling bending direction."},
        {EntityAttribute::MuscleMaxDistanceDeviation, "Maximum change of the connection length, relative to its initial value."},

        // Defender
        {EntityAttribute::DefenderMode,
         "Selects against which foreign cells the creature is protected. Only the affected cell itself and its directly connected cells count."
         "\n" ICON_FA_CHEVRON_RIGHT " Anti-attacker: reduces the energy that attacker cells can steal from the own creature."
         "\n" ICON_FA_CHEVRON_RIGHT " Anti-injector: increases the energy cost for injector cells that target the own creature."},

        // Reconnector
        {EntityAttribute::ReconnectorMode,
         "Selects the objects with which connections are established. A positive value in channel #0 creates a connection, a negative one removes existing "
         "connections."
         "\n" ICON_FA_CHEVRON_RIGHT " Solid: connections with solid particles."
         "\n" ICON_FA_CHEVRON_RIGHT " Free cell: connections with free cells."
         "\n" ICON_FA_CHEVRON_RIGHT " Creature: connections with cells of other creatures."},
        {EntityAttribute::ReconnectorRestrictToColors, "Connections are only established with objects of the selected customization colors."},
        {EntityAttribute::ReconnectorRestrictToLineage,
         "Restricts the connections to the relationship with the own creature."
         "\n" ICON_FA_CHEVRON_RIGHT " No: the lineage is not taken into account."
         "\n" ICON_FA_CHEVRON_RIGHT " Same lineage: connections only with creatures of the own lineage."
         "\n" ICON_FA_CHEVRON_RIGHT " Other lineage: connections only with creatures of a foreign lineage."},
        {EntityAttribute::ReconnectorMinNumCells,
         "Connections are only established with creatures that have at least this number of cells. Without a value there is no lower limit."},
        {EntityAttribute::ReconnectorMaxNumCells,
         "Connections are only established with creatures that have at most this number of cells. Without a value there is no upper limit."},

        // Detonator
        {EntityAttribute::DetonatorState,
         "State of the detonator."
         "\n" ICON_FA_CHEVRON_RIGHT " Ready: waiting for a trigger via channel #0."
         "\n" ICON_FA_CHEVRON_RIGHT " Activated: the countdown is running."
         "\n" ICON_FA_CHEVRON_RIGHT " Exploded: the detonator has already detonated and stays inactive."},
        {EntityAttribute::DetonatorCountdown, "Number of cycles until the explosion after the trigger. One cycle comprises 6 time steps."},

        // Digestor
        {EntityAttribute::DigestorEnergyConductivity,
         "Fraction of the raw energy that the cell passes on to connected digestor cells instead of converting it. Complementary to 'Energy conversion'."},
        {EntityAttribute::DigestorEnergyConversion,
         "Fraction of the raw energy that the cell converts into usable energy per cycle. Complementary to 'Energy conductivity'."},

        // Memory
        {EntityAttribute::MemoryMode,
         "Selects how the signal is stored."
         "\n" ICON_FA_CHEVRON_RIGHT " Signal delay: outputs the signal again after a fixed number of cycles."
         "\n" ICON_FA_CHEVRON_RIGHT " Signal recorder: records the signal into the buffer or plays it back, controlled by the sign of channel #0."
         "\n" ICON_FA_CHEVRON_RIGHT " Signal storage: addresses one buffer entry via the absolute value of channel #0 and reads or overwrites it."
         "\n" ICON_FA_CHEVRON_RIGHT " Signal integrator: outputs a running average of the signal."},
        {EntityAttribute::MemoryDelay, "Number of cycles by which the signal is delayed. It also determines the size of the ring buffer."},
        {EntityAttribute::MemoryReadOnly, "If enabled, the cell only reads from the signal buffer and never overwrites it."},
        {EntityAttribute::MemoryNewSignalWeight,
         "Weight of the current signal in the running average. At 0 the stored value is frozen, at 1 there is no smoothing."},
        {EntityAttribute::MemoryChannelMask,
         "Selects the channels that are overwritten with the stored values. The remaining channels are passed through unchanged."},
        {EntityAttribute::MemorySignalBuffer, "Opens the editor for the stored signal entries."},

        // Communicator
        {EntityAttribute::CommunicatorMode,
         "Selects the role in the communication. It only takes place between different creatures and requires a valid front direction on both sides."
         "\n" ICON_FA_CHEVRON_RIGHT " Send: transmits all own channels to the receivers in range. Requires a trigger via channel #0."
         "\n" ICON_FA_CHEVRON_RIGHT " Receive: takes over the channels of a sender. Channel #1 is converted into the own frame of reference."},
        {EntityAttribute::CommunicatorRange, "Radius within which receiver cells are addressed."},
        {EntityAttribute::CommunicatorOneway,
         "If enabled, only the receivers on one side are addressed, namely those opposite to the direction that is encoded in channel #1."},
        {EntityAttribute::CommunicatorRestrictToColors, "Only signals from senders with one of the selected customization colors are accepted."},
        {EntityAttribute::CommunicatorRestrictToLineage,
         "Restricts the reception to the relationship with the own creature."
         "\n" ICON_FA_CHEVRON_RIGHT " No: the lineage is not taken into account."
         "\n" ICON_FA_CHEVRON_RIGHT " Same lineage: only senders of the own lineage are accepted."
         "\n" ICON_FA_CHEVRON_RIGHT " Other lineage: only senders of a foreign lineage are accepted."},

        // Neural activity
        {EntityAttribute::NeuralSignal, "Value of the signal channel. It is the output of the neural network and the input for the connected cells."},
        {EntityAttribute::NeuralMemory,
         "Value of the memory neuron. It is retained across cycles and is only recalculated when the gate input exceeds the trigger threshold."},

        // Creature
        {EntityAttribute::CreatureId, "Unique 64 bit id of the creature."},
        {EntityAttribute::CreatureGeneration, "Number of times the genome has been passed on to an offspring."},
        {EntityAttribute::CreatureNumCells, "Number of cells that currently belong to the creature."},
        {EntityAttribute::CreatureLineageId,
         "Id of the lineage. Creatures with the same id count as related. A new id is assigned as soon as the accumulated mutations exceed the simulation "
         "parameter 'New lineage threshold'."},
        {EntityAttribute::CreatureMutationsInLineage, "Accumulated mutations since the current lineage was formed."},
        {EntityAttribute::CreatureMutationsTotal, "Accumulated mutations over the entire ancestry. This value is never reset."},
        {EntityAttribute::CreatureGenomeName, "Name of the genome from which the creature is built."},
        {EntityAttribute::CreatureResistanceToInjection, "If active, injector cells cannot overwrite the genome of this creature."},
        {EntityAttribute::CreatureApplyMetaMutations, "If active, the mutation rates stored in the genome are themselves subject to mutation."},
        {EntityAttribute::CreatureEditGenome, "Opens the genome of the creature in the genome editor."},
    });

    constexpr bool isComplete()
    {
        if (Descriptions.size() != static_cast<size_t>(EntityAttribute::Count)) {
            return false;
        }
        for (size_t index = 0; index < Descriptions.size(); ++index) {
            if (static_cast<size_t>(Descriptions.at(index).attribute) != index) {
                return false;
            }
        }
        return true;
    }
    static_assert(isComplete(), "Descriptions must contain every EntityAttribute exactly once and in the order of the enum.");
}

std::string EntityAttributeHelp::get(EntityAttribute attribute)
{
    return std::string(Descriptions.at(static_cast<size_t>(attribute)).description);
}
