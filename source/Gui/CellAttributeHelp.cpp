#include "CellAttributeHelp.h"

#include <array>
#include <string_view>

#include <Fonts/IconsFontAwesome5.h>

namespace
{
    struct Entry
    {
        CellAttribute attribute;
        std::string_view description;
    };

    constexpr auto Descriptions = std::to_array<Entry>({

        // Object
        {CellAttribute::ObjectId, "Unique 64 bit id of the object. It is assigned by the simulation and cannot be changed."},
        {CellAttribute::Position, "Position of the object in world coordinates."},
        {CellAttribute::Velocity, "Velocity of the object in world units per time step."},
        {CellAttribute::Stiffness, "Resistance of the object against deformation of its bonds."},
        {CellAttribute::Color, "Customization color of the object. Many simulation parameters can be defined per color."},
        {CellAttribute::Static, "A static object is immovable and immortal. It does not age and is not affected by forces."},
        {CellAttribute::Sticky, "If enabled, the object can spontaneously form new bonds on contact. It is sufficient if one of the two objects is sticky."},
        {CellAttribute::ObjectType,
         "Type of the object."
         "\n" ICON_FA_CHEVRON_RIGHT " Solid: inorganic rigid particle. It blocks the scan rays of sensor cells."
         "\n" ICON_FA_CHEVRON_RIGHT " Fluid: inorganic freely flowing particle without bonds."
         "\n" ICON_FA_CHEVRON_RIGHT " Free cell: organic substance without a genome. It can serves as food."
         "\n" ICON_FA_CHEVRON_RIGHT " Cell: cell of a creature with genome, neural network and cell type."},
        {CellAttribute::ConnectedId, "Id of the object that is attached by this connection."},
        {CellAttribute::ConnectionDistance, "Reference distance of the connection. A deviation of the actual distance generates restoring forces."},
        {CellAttribute::ConnectionRefAngle, "Reference angle between this connection and the previous one. A deviation generates restoring forces."},

        // Energy particle
        {CellAttribute::ParticleId, "Unique 64 bit id of the energy particle."},
        {CellAttribute::ParticleEnergy, "Energy of the particle. Cells can absorb it."},

        // Solid, fluid and free cell
        {CellAttribute::SolidEnergy, "Energy stored in the solid particle."},
        {CellAttribute::FluidEnergy, "Energy stored in the fluid particle."},
        {CellAttribute::FluidGlow, "Additional brightness of the fluid particle during rendering."},
        {CellAttribute::FreeCellEnergy,
         "Energy of the free cell. Below the minimum cell energy the cell decays."},
        {CellAttribute::FreeCellAge, "Age of the free cell in time steps."},

        // Cell
        {CellAttribute::CellUsableEnergy,
         "Energy that keeps the cell alive and can be use for creating new ones. Below the minimum cell energy the cell starts dying."},
        {CellAttribute::CellRawEnergy, "Unprocessed energy. Attacker cells gain raw energy while digestor cells convert it into usable energy."},
        {CellAttribute::CellFrontAngle,
         "Angle between the first connection and the front direction of the creature. It orients muscle, sensor and communicator cells."},
        {CellAttribute::CellAge, "Age of the cell in time steps."},
        {CellAttribute::CellState,
         "State of the cell in its life cycle."
         "\n" ICON_FA_CHEVRON_RIGHT " Ready: The cell is in normal operating mode. Neural networks and additional cell functions can be executed."
         "\n" ICON_FA_CHEVRON_RIGHT " Under construction: The cell belongs to an unfinished construction and is inactive."
         "\n" ICON_FA_CHEVRON_RIGHT " Being activated: Transitional state after the construction. The cell becomes ready in the next step."
         "\n" ICON_FA_CHEVRON_RIGHT " Dying: The cell decays with the probability given by the simulation parameter 'Cell death probability'."
         "\n" ICON_FA_CHEVRON_RIGHT " Instant dying: The cell is removed in the next time step."},
        {CellAttribute::CellNodeIndex, "Index of the genome node from which this cell was built."},
        {CellAttribute::CellParentNodeIndex, "Node index of the cell that built this cell."},
        {CellAttribute::CellGeneIndex, "Index of the gene from which this cell was built."},
        {CellAttribute::CellConcatenationIndex, "Index of the concatenation in which this cell was built."},
        {CellAttribute::CellBranchIndex, "Index of the branch in which this cell was built."},
        {CellAttribute::CellActivationTime,
         "Remaining time steps until the cell starts executing its function. As long as the value is greater than 0, the cell is inactive."},
        {CellAttribute::CellHeadCell,
         "Marks the head cell of the creature. Cells that lose connection to head cells die off."},
        {CellAttribute::CellType,
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
        {CellAttribute::CellHasConstructor, "If enabled, the cell can build other cells specified by a gene."},

        // Genome
        {CellAttribute::GenomeName, "Name of the genome. Every creature that is built from it inherits the name."},
        {CellAttribute::GenomeFrontAngle,
         "Angle between the first bond of the head cell and the front direction of the creature. From the head cell the front direction is propagated to all "
         "other cells."},
        {CellAttribute::GenomeResistanceToInjection, "If enabled, injector cells cannot overwrite the genome of creatures built from it."},
        {CellAttribute::GenomeApplyMetaMutations,
         "If enabled, the mutation rates of this genome are themselves mutated. The step size comes from the simulation parameters under 'Meta-mutations'."},
        {CellAttribute::GenomeMutationRatesEdit, "Opens the editor for all mutation rates. The rows below only list the rates that are currently active."},

        // Gene
        {CellAttribute::GeneName, "Name of the gene. It is only used for the display in the structure list."},
        {CellAttribute::GeneShape,
         "Determines the angles and the additional bonds with which the nodes of this gene are assembled. The angles of the first and the last node come from "
         "the nodes themselves."},
        {CellAttribute::GeneConnectionDistance, "Reference distance between two cells that are constructed one after another from this gene."},
        {CellAttribute::GeneStiffness,
         "Stiffness of all cells constructed from this gene. It determines how strongly they resist a deformation of their bonds."},
        {CellAttribute::GeneHomogeneousCellType, "If enabled, every constructed cell of this gene uses the cell type and its properties of the first node."},

        // Mutation rates. All mutations are applied exactly once per offspring, namely when its genome is passed on.
        {CellAttribute::MutationConnectionProbability, "Probability per node and bond that the weight of that bond in the neural network is changed."},
        {CellAttribute::MutationConnectionValueChangeSigma, "Standard deviation of the Gaussian change of the bond weight. The result is limited to [-2, 2]."},
        {CellAttribute::MutationNeuronProbability, "Probability per node and neuron that the neuron is changed."},
        {CellAttribute::MutationNeuronWeightChangeSigma,
         "Standard deviation of the Gaussian change of every input weight of the neuron. The result is limited to [-2, 2]."},
        {CellAttribute::MutationNeuronBiasChangeSigma, "Standard deviation of the Gaussian change of the bias of the neuron."},
        {CellAttribute::MutationNeuronActfnChangeProbability, "Probability that the activation function of the neuron is replaced by a different one."},
        {CellAttribute::MutationCellTypePropertiesProbability, "Probability per node that the properties of its cell type are changed."},
        {CellAttribute::MutationCellTypePropertiesValueChangeSigma,
         "Standard deviation of the Gaussian change of the numeric properties, relative to the value range of each property."},
        {CellAttribute::MutationCellTypePropertiesEnumChangeProbability, "Probability that a switch, a mode or a color selection of the cell type is changed."},
        {CellAttribute::MutationGeometryProbability, "Probability per gene that its geometry is changed."},
        {CellAttribute::MutationGeometryValueChangeSigma,
         "Standard deviation of the Gaussian change of the stiffness, relative to its value range. The connection distance is not mutated."},
        {CellAttribute::MutationGeometryEnumChangeProbability, "Probability that the shape generator of the gene is replaced by a different one."},
        {CellAttribute::MutationCellTypeModeProbability,
         "Probability per node that the mode of its cell type is switched to a different one. The mode-specific properties are reset to their default values; "
         "cell types without a mode stay unchanged."},
        {CellAttribute::MutationCellTypeProbability,
         "Probability per node that its cell type is replaced by a different one, with all properties reset to their default values. Void nodes are not "
         "affected. With the same probability the switch 'Homogeneous cell type' of the gene is toggled."},
        {CellAttribute::MutationCustomizationProbability,
         "Probability per genome that one used customization color is replaced everywhere by another one. Which replacements are permitted is defined by the "
         "simulation parameter 'Customization transition matrix'."},
        {CellAttribute::MutationVoidProbability,
         "Probability per node that it is toggled between void and a random other cell type. The first and the last node of a gene are not affected."},
        {CellAttribute::MutationExtendGeneProbability, "Probability per gene that a new node is appended at its beginning or at its end."},
        {CellAttribute::MutationAddNodeProbability, "Probability per insertion position that a new node is inserted there."},
        {CellAttribute::MutationTrimGeneProbability,
         "Probability per gene that its first or its last node is removed. The last remaining node of a gene is always kept."},
        {CellAttribute::MutationDeleteNodeProbability, "Probability per node that it is removed. The last remaining node of a gene is always kept."},
        {CellAttribute::MutationDuplicateGeneProbability,
         "Probability per gene that it is duplicated. Only genes that are referenced at least twice are duplicated; one of these references then points to the "
         "copy."},
        {CellAttribute::MutationDeleteGeneProbability,
         "Probability per gene that it is deleted. At least one gene is kept. A reference to a deleted gene switches off the referencing constructor and "
         "points a referencing injector to the first gene."},
        {CellAttribute::MutationCopyNodeSectionProbability,
         "Probability per gene that a contiguous section of its nodes is copied into a randomly chosen gene, which can also be the gene itself."},
        {CellAttribute::MutationMoveNodeSectionProbability,
         "Probability per gene that a contiguous section of its nodes is moved into a randomly chosen gene. At least one node stays behind."},
        {CellAttribute::MutationConstructorProbability, "Probability per node that the constructor properties of the node are changed."},
        {CellAttribute::MutationConstructorValueChangeSigma,
         "Standard deviation of the Gaussian change of the numeric constructor properties, relative to the value range of each property."},
        {CellAttribute::MutationConstructorEnumChangeProbability, "Probability that a switch or a selection of the constructor is changed."},
        {CellAttribute::MutationConstructorToggleProbability, "Probability that the constructor of the node is switched on or off."},

        // Genome node
        {CellAttribute::NodeAngle,
         "Angle of this cell relative to the bond to the previous cell. It is only evaluated for the first and the last node of a gene; for inner nodes the "
         "angle results from the selected shape."},
        {CellAttribute::NodeCustomization, "Customization color of the cell. Many simulation parameters are defined per color."},
        {CellAttribute::NodeConstructionGene, "Gene that this cell builds as a constructor. 'None' means that the cell has no constructor."},

        // Constructor
        {CellAttribute::ConstructorAutoTriggerInterval,
         "If set, the constructor triggers itself every n time steps, with a phase that differs per creature. Without a value it has to be triggered via "
         "channel #0."},
        {CellAttribute::ConstructorActivationTime,
         "Number of time steps for which a newly constructed cell is to remain inactive. Currently without effect: the engine does not evaluate this value."},
        {CellAttribute::ConstructorConstructionAngle,
         "Angle of the first constructed cell relative to the bond of the constructor cell. It is only evaluated for the first cell of the first concatenation "
         "in the first branch."},
        {CellAttribute::ConstructorProvideEnergy,
         "Determines where the energy for the construction comes from."
         "\n" ICON_FA_CHEVRON_RIGHT " Reduce cell energy: every constructed cell takes its energy from the constructor cell."
         "\n" ICON_FA_CHEVRON_RIGHT " Free: cells are built without energy cost. After the first completed offspring the setting falls back to 'Reduce cell "
         "energy'."},
        {CellAttribute::ConstructorReservedEnergy,
         "Energy reserve of the constructor cell. It is used up for the construction before the usable energy of the cell. External energy inflow is credited "
         "here."},
        {CellAttribute::ConstructorSeparation, "If enabled, the offspring detaches from the constructor cell after the last cell has been built."},
        {CellAttribute::ConstructorNumBranches, "Number of cell networks that the constructor builds next to each other from the same gene."},
        {CellAttribute::ConstructorNumConcatenations, "How often the gene is built in a row, each attached to the previous one."},
        {CellAttribute::ConstructorGeneIndex, "Index of the gene that the constructor builds."},

        // Depot
        {CellAttribute::DepotStorageLimit, "Maximum usable energy that the depot can store."},
        {CellAttribute::DepotInitialStoredEnergy,
         "Energy that the depot already contains when it is constructed. The constructor has to provide this energy in addition."},
        {CellAttribute::DepotStoredEnergy, "Usable energy that is currently stored in the depot."},

        // Sensor
        {CellAttribute::SensorAutoTrigger,
         "If enabled, the sensor scans in every cycle. Otherwise it only scans when channel #0 exceeds the trigger threshold."},
        {CellAttribute::SensorTagForAttackers,
         "If enabled, attacker cells of the same creature in 'Creature' mode may use the last match of this sensor as a target."},
        {CellAttribute::SensorMode,
         "Selects what the sensor scans for."
         "\n" ICON_FA_CHEVRON_RIGHT " Detect energy: searches for accumulations of energy particles."
         "\n" ICON_FA_CHEVRON_RIGHT " Detect solid: searches for solid particles."
         "\n" ICON_FA_CHEVRON_RIGHT " Detect free cell: searches for accumulations of free cells."
         "\n" ICON_FA_CHEVRON_RIGHT " Detect creature: searches for cells of other creatures."},
        {CellAttribute::SensorEnergyMinDensity, "Minimum density of energy particles that is required for a match. The value lies between 0 and 1."},
        {CellAttribute::SensorFreeCellMinDensity, "Minimum density of free cells that is required for a match. The value lies between 0 and 1."},
        {CellAttribute::SensorMinRange, "Objects that are closer than this distance are not detected."},
        {CellAttribute::SensorMaxRange,
         "Objects that are farther away than this distance are not detected. The simulation parameter 'Sensor radius' limits the range additionally."},
        {CellAttribute::SensorRestrictToColors, "Only objects with one of the selected customization colors are detected."},
        {CellAttribute::SensorRestrictToLineage,
         "Restricts the detection to the relationship with the own creature."
         "\n" ICON_FA_CHEVRON_RIGHT " No: the lineage is not taken into account."
         "\n" ICON_FA_CHEVRON_RIGHT " Same lineage: only creatures of the own lineage are detected."
         "\n" ICON_FA_CHEVRON_RIGHT " Other lineage: only creatures of a foreign lineage are detected."},
        {CellAttribute::SensorMinNumCells, "Only creatures with at least this number of cells are detected. Without a value there is no lower limit."},
        {CellAttribute::SensorMaxNumCells, "Only creatures with at most this number of cells are detected. Without a value there is no upper limit."},

        // Generator
        {CellAttribute::GeneratorAdditive, "If enabled, the generated value is added to channel #0 instead of overwriting it."},
        {CellAttribute::GeneratorMinValue, "Lower level of the generated signal."},
        {CellAttribute::GeneratorMaxValue, "Upper level of the generated signal."},
        {CellAttribute::GeneratorTimeOffset, "Shifts the phase of the signal by this number of time steps."},
        {CellAttribute::GeneratorMode,
         "Shape of the generated signal."
         "\n" ICON_FA_CHEVRON_RIGHT " Square signal: the maximum value is output in the first half of the period and the minimum value in the second half."
         "\n" ICON_FA_CHEVRON_RIGHT " Sawtooth signal: the output rises linearly from the minimum to the maximum value over the period."},
        {CellAttribute::GeneratorPeriod, "Length of one cycle in time steps."},

        // Attacker
        {CellAttribute::AttackerMode,
         "Selects the targets of the attack. The stolen energy is added to the own raw energy."
         "\n" ICON_FA_CHEVRON_RIGHT " Free cell: steals energy from free cells within the attack radius."
         "\n" ICON_FA_CHEVRON_RIGHT " Creature: steals energy from cells of other creatures. Only creatures that a sensor cell of the same creature has "
         "detected and tagged are attacked."},
        {CellAttribute::AttackerRestrictToColors, "Only free cells with one of the selected customization colors are attacked."},

        // Injector
        {CellAttribute::InjectorGeneIndex, "Gene index that is set in the constructor of the injected cell. It determines which gene is built there."},

        // Muscle
        {CellAttribute::MuscleMode,
         "Selects how the muscle generates movement. Bending and crawling act on the first bond."
         "\n" ICON_FA_CHEVRON_RIGHT " Auto bending: bends back and forth on its own between the angle limits. Channel #1 sets the target direction."
         "\n" ICON_FA_CHEVRON_RIGHT " Manual bending: the bending direction follows the sign of channel #0."
         "\n" ICON_FA_CHEVRON_RIGHT " Angle bending: bends towards the direction given in channel #1 and turns the front direction along with it."
         "\n" ICON_FA_CHEVRON_RIGHT " Auto crawling: changes the bond length back and forth on its own."
         "\n" ICON_FA_CHEVRON_RIGHT " Manual crawling: the change of the bond length follows the sign of channel #0."
         "\n" ICON_FA_CHEVRON_RIGHT " Direct movement: accelerates the cell directly in the direction given in channel #1."},
        {CellAttribute::MuscleMaxAngleDeviation, "Maximum deviation from the initial angle, as a fraction of the available angular range."},
        {CellAttribute::MuscleForwardBackwardRatio,
         "Distributes the speed between the forward and the backward stroke. Since the faster stroke generates more thrust, this determines the direction of "
         "movement. At 0.5 both strokes cancel each other out."},
        {CellAttribute::MuscleAttractionRepulsionRatio, "Distributes the speed between the attracting and the repelling bending direction."},
        {CellAttribute::MuscleMaxDistanceDeviation, "Maximum change of the bond length, relative to its initial value."},

        // Defender
        {CellAttribute::DefenderMode,
         "Selects against which foreign cells the creature is protected. Only the affected cell itself and its directly connected cells count."
         "\n" ICON_FA_CHEVRON_RIGHT " Anti-attacker: reduces the energy that attacker cells can steal from the own creature."
         "\n" ICON_FA_CHEVRON_RIGHT " Anti-injector: increases the energy cost for injector cells that target the own creature."},

        // Reconnector
        {CellAttribute::ReconnectorMode,
         "Selects the objects with which bonds are established. A positive value in channel #0 creates a bond, a negative one removes existing bonds."
         "\n" ICON_FA_CHEVRON_RIGHT " Solid: bonds with solid particles."
         "\n" ICON_FA_CHEVRON_RIGHT " Free cell: bonds with free cells."
         "\n" ICON_FA_CHEVRON_RIGHT " Creature: bonds with cells of other creatures."},
        {CellAttribute::ReconnectorRestrictToColors, "Bonds are only established with objects of the selected customization colors."},
        {CellAttribute::ReconnectorRestrictToLineage,
         "Restricts the bonds to the relationship with the own creature."
         "\n" ICON_FA_CHEVRON_RIGHT " No: the lineage is not taken into account."
         "\n" ICON_FA_CHEVRON_RIGHT " Same lineage: bonds only with creatures of the own lineage."
         "\n" ICON_FA_CHEVRON_RIGHT " Other lineage: bonds only with creatures of a foreign lineage."},
        {CellAttribute::ReconnectorMinNumCells,
         "Bonds are only established with creatures that have at least this number of cells. Without a value there is no lower limit."},
        {CellAttribute::ReconnectorMaxNumCells,
         "Bonds are only established with creatures that have at most this number of cells. Without a value there is no upper limit."},

        // Detonator
        {CellAttribute::DetonatorState,
         "State of the detonator."
         "\n" ICON_FA_CHEVRON_RIGHT " Ready: waiting for a trigger via channel #0."
         "\n" ICON_FA_CHEVRON_RIGHT " Activated: the countdown is running."
         "\n" ICON_FA_CHEVRON_RIGHT " Exploded: the detonator has already detonated and stays inactive."},
        {CellAttribute::DetonatorCountdown, "Number of cycles until the explosion after the trigger. One cycle comprises 6 time steps."},

        // Digestor
        {CellAttribute::DigestorEnergyConductivity,
         "Fraction of the raw energy that the cell passes on to connected digestor cells instead of converting it. Complementary to 'Energy conversion'."},
        {CellAttribute::DigestorEnergyConversion,
         "Fraction of the raw energy that the cell converts into usable energy per cycle. Complementary to 'Energy conductivity'."},

        // Memory
        {CellAttribute::MemoryMode,
         "Selects how the signal is stored."
         "\n" ICON_FA_CHEVRON_RIGHT " Signal delay: outputs the signal again after a fixed number of cycles."
         "\n" ICON_FA_CHEVRON_RIGHT " Signal recorder: records the signal into the buffer or plays it back, controlled by the sign of channel #0."
         "\n" ICON_FA_CHEVRON_RIGHT " Signal storage: addresses one buffer entry via the absolute value of channel #0 and reads or overwrites it."
         "\n" ICON_FA_CHEVRON_RIGHT " Signal integrator: outputs a running average of the signal."},
        {CellAttribute::MemoryDelay, "Number of cycles by which the signal is delayed. It also determines the size of the ring buffer."},
        {CellAttribute::MemoryReadOnly, "If enabled, the cell only reads from the signal buffer and never overwrites it."},
        {CellAttribute::MemoryNewSignalWeight,
         "Weight of the current signal in the running average. At 0 the stored value is frozen, at 1 there is no smoothing."},
        {CellAttribute::MemoryChannelMask,
         "Selects the channels that are overwritten with the stored values. The remaining channels are passed through unchanged."},
        {CellAttribute::MemorySignalBuffer, "Opens the editor for the stored signal entries."},

        // Communicator
        {CellAttribute::CommunicatorMode,
         "Selects the role in the communication. It only takes place between different creatures and requires a valid front direction on both sides."
         "\n" ICON_FA_CHEVRON_RIGHT " Send: transmits all own channels to the receivers in range. Requires a trigger via channel #0."
         "\n" ICON_FA_CHEVRON_RIGHT " Receive: takes over the channels of a sender. Channel #1 is converted into the own frame of reference."},
        {CellAttribute::CommunicatorRange, "Radius within which receiver cells are addressed."},
        {CellAttribute::CommunicatorOneway,
         "If enabled, only the receivers on one side are addressed, namely those opposite to the direction that is encoded in channel #1."},
        {CellAttribute::CommunicatorRestrictToColors, "Only signals from senders with one of the selected customization colors are accepted."},
        {CellAttribute::CommunicatorRestrictToLineage,
         "Restricts the reception to the relationship with the own creature."
         "\n" ICON_FA_CHEVRON_RIGHT " No: the lineage is not taken into account."
         "\n" ICON_FA_CHEVRON_RIGHT " Same lineage: only senders of the own lineage are accepted."
         "\n" ICON_FA_CHEVRON_RIGHT " Other lineage: only senders of a foreign lineage are accepted."},

        // Neural activity
        {CellAttribute::NeuralSignal, "Value of the signal channel. It is the output of the neural network and the input for the connected cells."},
        {CellAttribute::NeuralMemory,
         "Value of the memory neuron. It is retained across cycles and is only recalculated when the gate input exceeds the trigger threshold."},

        // Creature
        {CellAttribute::CreatureId, "Unique 64 bit id of the creature."},
        {CellAttribute::CreatureGeneration, "Number of times the genome has been passed on to an offspring."},
        {CellAttribute::CreatureNumCells, "Number of cells that currently belong to the creature."},
        {CellAttribute::CreatureLineageId,
         "Id of the lineage. Creatures with the same id count as related. A new id is assigned as soon as the accumulated mutations exceed the simulation "
         "parameter 'New lineage threshold'."},
        {CellAttribute::CreatureMutationsInLineage, "Accumulated mutations since the current lineage was formed."},
        {CellAttribute::CreatureMutationsTotal, "Accumulated mutations over the entire ancestry. This value is never reset."},
        {CellAttribute::CreatureGenomeName, "Name of the genome from which the creature is built."},
        {CellAttribute::CreatureResistanceToInjection, "If active, injector cells cannot overwrite the genome of this creature."},
        {CellAttribute::CreatureApplyMetaMutations, "If active, the mutation rates stored in the genome are themselves subject to mutation."},
        {CellAttribute::CreatureEditGenome, "Opens the genome of the creature in the genome editor."},
    });

    constexpr bool isComplete()
    {
        if (Descriptions.size() != static_cast<size_t>(CellAttribute::Count)) {
            return false;
        }
        for (size_t index = 0; index < Descriptions.size(); ++index) {
            if (static_cast<size_t>(Descriptions.at(index).attribute) != index) {
                return false;
            }
        }
        return true;
    }
    static_assert(isComplete(), "Descriptions must contain every CellAttribute exactly once and in the order of the enum.");
}

std::string CellAttributeHelp::get(CellAttribute attribute)
{
    return std::string(Descriptions.at(static_cast<size_t>(attribute)).description);
}
