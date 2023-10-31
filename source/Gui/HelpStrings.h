#pragma once

#include <string>
#include <vector>

#include "Fonts/IconsFontAwesome5.h"

#include "EngineInterface/CellFunctionConstants.h"

namespace Const
{
    std::string const GeneralInformation =
        "Please make sure that:\n\n1) You have an NVIDIA graphics card with compute capability 6.0 or higher (for example "
        "GeForce 10 series).\n\n2) You have the latest NVIDIA graphics driver installed.\n\n3) The name of the "
        "installation directory (including the parent directories) should not contain non-English characters. If this is not fulfilled, "
        "please re-install ALIEN to a suitable directory. Do not move the files manually. If you use Windows, make also sure that you install ALIEN with a "
        "Windows user that contains no non-English characters. If this is not the case, a new Windows user could be created to solve this problem.\n\n4) ALIEN needs "
        "write access to its own "
        "directory. This should normally be the case.\n\n5) If you have multiple graphics cards, please check that your primary monitor is "
        "connected to the CUDA-powered card. ALIEN uses the same graphics card for computation as well as rendering and chooses the one "
        "with the highest compute capability.\n\n6) If you possess both integrated and dedicated graphics cards, please ensure that the alien-executable is "
        "configured to use your high-performance graphics card. On Windows you need to access the 'Graphics settings,' add 'alien.exe' to the list, click "
        "'Options,' and choose 'High performance'.\n\nIf these conditions are not met, ALIEN may crash unexpectedly.\n\n"
        "If the conditions are met and the error still occurs, please start ALIEN with the command line parameter '-d', try to reproduce the error and "
        "then create a GitHub issue on https://github.com/chrxh/alien/issues where the log.txt is attached.";

    std::string const NeuronTooltip =
        "This function equips the cell with a small network of 8 neurons with 8x8 configurable weights, 8 bias values and activation functions. It processes "
        "the input from channel #0 to #7 and provides the output to those channels. More precisely, the output of each neuron calculates as\noutput_j := "
        "sigma(sum_i (input_i * weight_ji) + bias_j),\nwhere sigma stands for the activation function (different choices are available).";

    std::string const TransmitterTooltip =
        "Transmitter cells are designed to transport energy. This is important, for example, to supply constructor cells with energy or to "
        "support attacked cells. The energy transport works as follows: A part of the excess energy of the own cell and the directly connected "
        "cells is collected and transferred to other cells in the vicinity. A cell has excess energy when it exceeds a defined normal value (see "
        "simulation parameter 'Normal energy' in 'Cell life cycle'). Transmitter cells do not need an activation but they also transport the "
        "activity states from input.";

    std::string const ConstructorTooltip =
        "A constructor cell builds a cell network according to a contained genome. The construction process takes place cell by "
        "cell, where energy is required for each new cell. Once a new cell is generated, it is connected to the already constructed "
        "cell network.\n\n" ICON_FA_CHEVRON_RIGHT " Input channel #0: abs(value) > threshold activates constructor (only necessary in "
        "'Manual' mode)\n\n" ICON_FA_CHEVRON_RIGHT " Output channel #0: 0 (could not constructor next cell, e.g. no energy, required "
        "connection check failed, completeness check failed), 1 (next cell construction successful)";

    std::string const SensorTooltip =
        "Sensor cells scan their environment for concentrations of cells of a certain color and provide distance and angle to the "
        "closest match.\n\n" ICON_FA_CHEVRON_RIGHT " Input channel #0: abs(value) > threshold activates sensor\n\n" ICON_FA_CHEVRON_RIGHT " Output channel #0: "
        "0 (no match) or 1 (match)\n\n" ICON_FA_CHEVRON_RIGHT " Output channel #1: density of the last match\n\n" ICON_FA_CHEVRON_RIGHT " Output channel #2: distance "
        "of the last match (0 = far away, 1 = close)\n\n" ICON_FA_CHEVRON_RIGHT " Output channel #3: angle of the last match";

    std::string const NerveTooltip =
        "By default, a nerve cell forwards activity states by receiving activity as input from connected cells (and summing it if "
        "there are multiple cells) and directly providing it as output to other cells. Independently of this, one can specify "
        "that it also generates an activity pulse in channel #0 at regular intervals. This can be used to trigger other sensor cells, "
        "attacker cells, etc.";

    std::string const AttackerTooltip =
        "An attacker cell attacks surrounding cells from other cell networks (with different creature id) by stealing energy from "
        "them. The gained energy is then distributed in the own cell network.\n\n" ICON_FA_CHEVRON_RIGHT " Input channel #0: abs(value) > threshold activates "
        "attacker\n\n" ICON_FA_CHEVRON_RIGHT " Output channel #0: a value which is proportional to the gained energy";

    std::string const InjectorTooltip =
        "Injector cells can override the genome of other constructor or injector cells by their own. To do this, they need to be activated, remain in "
        "close proximity to the target cell for a certain minimum duration, and, in the case of a target constructor cell, its construction process "
        "must not have started yet.\n\n" ICON_FA_CHEVRON_RIGHT " Input channel #0: abs(value) > threshold activates injector\n\n" ICON_FA_CHEVRON_RIGHT
        " Output channel #0: 0 (no cells found) or 1 (injection in process or completed)";

    std::string const MuscleTooltip =
        "Muscle cells can perform different movements and deformations based on input and configuration.\n\n" ICON_FA_CHEVRON_RIGHT " Input channel "
        "#0: The strength of the movement, bending or expansion/contraction. A negative sign corresponds to the opposite "
        "action.\n\n" ICON_FA_CHEVRON_RIGHT " Input channel #1: This channel is solely utilized for acceleration due to bending. If the sign of channel #1 "
        "differs from the sign of channel #0, no acceleration will be obtained during the bending process.\n\n " ICON_FA_CHEVRON_RIGHT
        " Input channel #3: This channel is used for muscles in movement mode. It allows to determine the relative angle of the movement. A value of -0.5 "
        "correspond to -180 deg and +0.5 to +180 deg.";

    std::string const DefenderTooltip =
        "A defender cell does not need to be activated. Its presence reduces the strength of an enemy attack involving attacker "
        "cells or extends the injection duration for injector cells.";

    std::string const ReconnectorTooltip =
        "A reconnector cell can make or break a cell connection to an other cell (with a different creature id) with a specified color. \n\n" ICON_FA_CHEVRON_RIGHT
        " Input channel #0: value > threshold triggers creation of a bond to a cell in the vicinity, value < -threshold triggers destruction of a bond\n\n" ICON_FA_CHEVRON_RIGHT
        " Output channel #0: 0 (no connection created/removed) or 1 (connection created/removed)";

    std::string const DetonatorTooltip = "A detonator cell will be activated if it receives an input on channel #0 with abs(value) > threshold. Then its counter "
                                         "is decreasing after each executing until it reaches 0. After that the detonator cell will explode and the surrounding cells are highly accelerated.";

    std::string const CellFunctionTooltip =
        "Cells can possess a specific function that enables them to, for example, perceive their environment, process information, or "
        "take action. All cell functions have in common that they obtain the input from connected cells whose execution number matches the input "
        "execution number of the current cell. For this purpose, each channel from #0 to #7 of those cells is summed and the result is written "
        "to the channel from #0 to #7 of the current cell. In particular, if there is only one input cell, its activity is simply forwarded. After "
        "the execution of a cell function, some channels will be then overriden by the output of the corresponding cell function.\n\nIMPORTANT: If "
        "you choose a cell function, this tooltip will be updated to provide more specific information. ";

    std::string const GenomeColorTooltip =
        "This property defines the color of the cell. It is not just a visual marker. On the one hand, the cell color can be used to define own types of cells "
        "that are subject to different rules. For this purpose, the simulation parameters can be specified depending on the color. For example, one could "
        "define that green cells are particularly good at absorbing energy particles, while other cell colors are better at attacking foreign cells.\nOn the "
        "other hand, cell color also plays a role in perception. Sensor cells are dedicated to a specific color and can only detect the corresponding cells.";

    std::string const GenomeAngleTooltip =
        "The angle between the predecessor and successor cell can be specified here. Please note that the shown angle here is shifted "
        "by 180 degrees for convenience. In other words, a value of 0 actually corresponds to an angle of 180 degrees, i.e. a straight "
        "segment.";

    std::string const GenomeEnergyTooltip =
        "The energy that the cell should receive after its creation. The larger this value is, the more energy the constructor cell must expend "
        "to create it.";

    std::string const GenomeExecutionNumberTooltip =
        "The functions of cells can be executed in a specific sequence determined by this number. The values are limited between 0 and 5 and "
        "follow a modulo 6 logic. For example, a cell with an execution number of 0 will be executed at time points 0, 6, 12, 18, etc. A cell "
        "with an execution number of 1 will be shifted by one, i.e. executed at 1, 7, 13, 19, etc. This time offset enables the orchestration "
        "of cell functions. A muscle cell, for instance, requiring input from a neuron cell, should then be executed one time step later.";

    std::string const GenomeInputExecutionNumberTooltip =
        "A functioning organism requires cells to collaborate. This can involve sensor cells that perceive the environment, neuron cells that "
        "process information, muscle cells that perform movements, and so on. These various cell functions often require input and produce an "
        "output. Both input and output are based on the cell's activity states. The process for updating is performed in two steps:\n\n1) When a "
        "cell function is executed, the activity states are first updated. This involves reading the activity states of all connected cells "
        "whose 'execution number' matches the specified 'input execution number', summing them up, and then setting the result to the "
        "activity states for the considered cell.\n\n2) The cell function is executed and can use the cell's activity states as input. "
        "The output is used to update the activity states again.\n\nSetting an 'input execution number' is optional. If none is set, the cell can "
        "receive no input.";

    std::string const GenomeBlockOutputTooltip =
        "Activating this toggle, the cell's output can be locked, preventing any other cell from utilizing it as input.";

    std::string const GenomeRequiredConnectionsTooltip =
        "By default, cells in the genome sequence are automatically connected to all neighboring cells belonging to the same genome when they "
        "are created. However, this can pose a challenge because the constructed cells need time to fold into their desired positions. If the "
        "current spatial location of the constructor cell is unfavorable, the newly formed cell might not be connected to the desired cells, "
        "for instance, due to being too far away. An better approach would involve delaying the construction process until a desired number of "
        "neighboring cells from the same genome are in the direct vicinity. This number of cells can be optionally set here.\nIt is important "
        "to note that the direct predecessor cell is not counted for the 'required connections.' For example, a value of 2 means that the cell to be "
        "constructed will only be created when there are at least 2 already constructed cells (excluding the predecessor cell) available for "
        "potential connections. If the condition is not met, the construction process is postponed.";

    std::string const GenomeNeuronActivationFunctionTooltip =
        "The activation function is a mapping which will be applied to the accumulated value from all inputs channels"
        " considering the weights and bias in order to calculate the neuron's output, i.e., output_j = sigma(sum_i (input_i * weight_ji) + bias_j), where sigma"
        " denotes the activation function. The following choices for sigma are available:\n\n" ICON_FA_CHEVRON_RIGHT
        " Sigmoid(x) := 2 / (1 + exp(x)) - 1\n\n" ICON_FA_CHEVRON_RIGHT " Binary step(x) := 1 if x >= 0 and 0 if x < 0\n\n" ICON_FA_CHEVRON_RIGHT
        " Identity(x) := x\n\n" ICON_FA_CHEVRON_RIGHT " Abs(x) := x if x >= 0 and -x if x < 0\n\n" ICON_FA_CHEVRON_RIGHT " Gaussian(x) := exp(-2 * x * x)";

    std::string const GenomeNeuronWeightAndBiasTooltip =
        "Each neuron has 8 input channels and produces an output by the formula output_j = sigma((sum_i (input_i * weight_ji) + "
        "bias_j), where sigma denotes the activation function.";

    std::string const GenomeTransmitterEnergyDistributionTooltip =
        "There are two ways to control the energy distribution, which is set "
        "here:\n\n" ICON_FA_CHEVRON_RIGHT " Connected cells: "
        "In this case the energy will be distributed evenly across all connected and connected-connected cells.\n\n" ICON_FA_CHEVRON_RIGHT
        " Transmitters and constructors: "
        "Here, the energy will be transferred to spatially nearby constructors or other transmitter cells within the same cell "
        "network. If multiple such transmitter cells are present at certain distances, energy can be transmitted over greater distances, "
        "for example, from attacker cells to constructor cells.";

    std::string const GenomeConstructorActivationModeTooltip =
        "There are 2 modes available for controlling constructor cells:\n\n" ICON_FA_CHEVRON_RIGHT " Manual: The construction process is only triggered when "
        "there is activity in channel #0.\n\n" ICON_FA_CHEVRON_RIGHT " Automatic: The construction process is automatically triggered at regular intervals. "
        "Activity in channel #0 is not necessary.\n\n In both cases, if there is not enough energy available for the cell being "
        "created, the construction process will pause until the next triggering.";

    std::string const GenomeConstructorIntervalTooltip =
        "This value specifies the time interval for automatic triggering of the constructor cell. It is given in multiples "
        "of 6 (which is a complete execution cycle). This means that a value of 1 indicates that the constructor cell will be activated "
        "every 6 time steps.";

    std::string const GenomeConstructorOffspringActivationTime =
        "When a new cell network has been fully constructed by a constructor cell, one can define the time steps until activation. Before activation, the cell "
        "network is in a dormant state. This is especially useful when the offspring should not become active immediately, for example, to prevent it from "
        "attacking its creator.";

    std::string const GenomeConstructorAngle1Tooltip =
        "By default, when the constructor cell initiates a new construction, the new cell is created in the area with the most available "
        "space. This angle specifies the deviation from that rule.";

    std::string const GenomeConstructorAngle2Tooltip =
        "This value determines the angle from the last constructed cell to the second-last constructed cell and the constructor cell. The "
        "effects can be best observed in the preview of the genome editor.";

    std::string const GenomeSensorModeTooltip =
        "Sensors can operate in 2 modes:\n\n" ICON_FA_CHEVRON_RIGHT " Scan vicinity: In this mode, the entire nearby area is scanned (typically "
        "within a radius of several 100 units). The scan radius can be adjusted via a simulation parameter (see 'Range' in the sensor "
        "settings).\n\n" ICON_FA_CHEVRON_RIGHT " Scan specific direction: In this mode, the scanning process is restricted to a particular direction. The "
        "direction is specified as an angle.";

    std::string const GenomeSensorScanAngleTooltip =
        "The angle in which direction the scanning process should take place can be determined here. An angle of 0 means that the "
        "scan will be performed in the direction derived from the input cell (the cell from which the activity input originates) "
        "towards the sensor cell.";

    std::string const GenomeSensorScanColorTooltip = "Specifies the color of the cells to search for.";

    std::string const GenomeSensorMinDensityTooltip =
        "The minimum density to search for a cell concentration of a specific color. This value ranges between 0 and 1. It controls the "
        "sensitivity of the sensor. Typically, very few cells of the corresponding color are already detected with a value of 0.1.";

    std::string const GenomeNerveGeneratePulsesTooltip = "If enabled, an activity pulse in channel #0 will be generated at regular time intervals.";

    std::string const GenomeNervePulseIntervalTooltip =
        "The intervals between two pulses can be set here. It is specified in cycles, which corresponds to 6 time steps each.";

    std::string const GenomeNerveAlternatingPulsesTooltip =
        "By default, the generated pulses consist of a positive value in channel #0. When 'Alternating pulses' is enabled, the "
        "sign of this value alternates at specific time intervals. This can be used, for example, to easily create activity "
        "signals for back-and-forth movements or bending in muscle cells.";

    std::string const GenomeNervePulsesPerPhaseTooltip = "This value indicates the number of pulses until the sign will be changed in channel #0.";

    std::string const GenomeAttackerEnergyDistributionTooltip =
        "Attacker cells can distribute the acquired energy through two different methods. The energy distribution is analogous to "
        "transmitter cells. \n\n" ICON_FA_CHEVRON_RIGHT " Connected cells: In this case the energy will be distributed evenly across all "
        "connected and connected-connected cells.\n\n" ICON_FA_CHEVRON_RIGHT
        " Transmitters and constructors: Here, the energy will be transferred to spatially nearby constructors or other transmitter cells "
        "within the same cell network. If multiple such transmitter cells are present at certain distances, energy can be transmitted "
        "over greater distances, for example, from attacker cells to constructor cells.";

    std::string const GenomeInjectorModeTooltip = ICON_FA_CHEVRON_RIGHT
        " Only empty cells: Only cells which possess an empty genome can be infected. This mode is useful when an organism wants to "
        "inject its genome into another own constructor cell (e.g. to build a spore). In this mode the injection process does not take any "
        "time.\n\n" ICON_FA_CHEVRON_RIGHT " All Cells: In this mode there are no restrictions, e.g. any other constructor or injector cell can be infected. "
        "The duration of the injection process depends on the simulation parameter 'Injection time'.";

    std::string const GenomeMuscleModeTooltip = ICON_FA_CHEVRON_RIGHT
        " Movement: Results in movement in the direction (or counter-direction) determined by the path from the "
        "input cell to the muscle cell.\n\n" ICON_FA_CHEVRON_RIGHT " Expansion and contraction: Causes an elongation (or contraction) of the "
        "reference distance to the input cell.\n\n" ICON_FA_CHEVRON_RIGHT " Bending: Increases (or decreases) the angle between the muscle "
        "cell, input cell, and the nearest connected cell clockwise from the muscle cell.";

    std::string const GenomeDefenderModeTooltip =
        ICON_FA_CHEVRON_RIGHT " Anti-attacker: reduces the attack strength of an enemy attacker cell\n\n" ICON_FA_CHEVRON_RIGHT
                              "Anti-injector: increases the injection duration of an enemy injector cell";

    std::string const GenomeReconnectorTargetColorTooltip = "Specifies the color of the cells where connections are to be established or destroyed.";

    std::string const DetonatorStateTooltip =
        ICON_FA_CHEVRON_RIGHT " Ready: The detonator cell waits for input on channel #0. If abs(value) > threshold, the detonator will be activated.\n\n"
        ICON_FA_CHEVRON_RIGHT " Activated: The countdown is decreased until 0 each time the detonator is executed. If the countdown is 0, the detonator will explode.\n\n"
        ICON_FA_CHEVRON_RIGHT " Exploded: The detonator is already exploded.";

    std::string const GenomeDetonatorCountdownTooltip = "The countdown specifies the cycles (in 6 time steps) until the detonator will explode.";

    std::string const SubGenomeTooltip =
        "If a constructor or injector cell is encoded in a genome, that cell can itself contain another genome. This sub-genome can "
        "describe additional body parts or branching of the creature, for instance. Furthermore, sub-genomes can in turn possess further "
        "sub-sub-genomes, etc. To insert a sub-genome here by clicking on 'Paste', one must have previously copied one to the clipboard. "
        "This can be done using the 'Copy genome' button in the toolbar. This action copies the entire genome from the current tab to "
        "the clipboard. If you want to create self-replication, you must not insert a sub-genome; instead, you switch it to the "
        "'self-copy' mode. In this case, the constructor's sub-genome refers to its superordinate genome.";

    std::string const GenomeGeometryTooltip =
        "A genome describes a network of connected cells. On the one hand, there is the option to select a pre-defined geometry (e.g. "
        "triangle or hexagon). Then, the cells encoded in the genome are generated along this geometry and connected together "
        "appropriately. On the other hand, it is also possible to define custom geometries by setting an angle between predecessor and "
        "successor cells for each cell (except for the first and last in the sequence).";

    std::string const GenomeConnectionDistanceTooltip =
        "The spatial distance between each cell and its predecessor cell in the genome sequence is determined here.";

    std::string const GenomeStiffnessTooltip = "This value sets the stiffness for the entire encoded cell network. The stiffness determines the amount of "
                                               "force generated to push the cell network to its reference configuration.";

    std::string const GenomeAngleAlignmentTooltip =
        "Triples of connected cells within a network have specific spatial angles relative to each other. These angles "
        "are guided by the reference angles encoded in the cells. With this setting, it is optionally possible to specify that the reference angles must only "
        "be multiples of certain values. This allows for greater stability of the created networks, as the angles would otherwise be more susceptible to "
        "external influences. Choosing 60 degrees is recommended here, as it allows for the accurate representation of most geometries.";

    std::string const GenomeMultipleConstructionsTooltip =
        "This flag specifies whether the construction described by the genome (repetitions included) should be built multiple times or not.";

    std::string const GenomeRepetitionsPerConstructionTooltip =
        "This value specifies how many times the cell network described in the genome should be concatenated for each construction. For a value greater "
        "than 1, the cell network geometry has to fulfill certain requirements (e.g. rectangle, hexagon, loop and lolli geometries are not suitable for concatenation).";

    std::string const GenomeConcatenationAngle1 =
        "This value describes the angle between two concatenated cell networks viewed from the first cell of the subsequent cell network.";

    std::string const GenomeConcatenationAngle2 =
        "This value describes the angle between two concatenated cell networks viewed from the last cell of the previous cell network.";

    std::string const GenomeSeparationConstructionTooltip =
        "Here, one can configure whether the encoded cell network in the genome should be detached from the constructor cell once it has been "
        "fully constructed. Disabling this property is useful for encoding growing structures (such as plant-like species) or creature body "
        "parts.";

    std::string const CellEnergyTooltip = "The amount of internal energy of the cell. The cell undergoes decay when its energy falls below a critical "
                                          "threshold (refer to the 'Minimum energy' simulation parameter).";

    std::string const CellStiffnessTooltip =
        "The stiffness determines the amount of force generated after a displacement to push the cell (network) to its reference configuration.";

    std::string const CellMaxConnectionTooltip = "The maximum number of bonds a cell can form with other cells.";

    std::string const CellIndestructibleTooltip =
        "When a cell is set as indestructible, it becomes immortal, resistant to external forces, but still capable of linear movement. Furthermore, unconnected "
        "normal cells and energy particles bounce off from indestructible ones.";

    std::string const CellReferenceDistanceTooltip =
        "The reference distance defines the distance at which no forces act between two connected cells. If the actual distance is greater than the reference "
        "distance, the cells attract each other. If it is smaller, they repel.";

    std::string const CellReferenceAngleTooltip =
        "The reference angle defines an angle between two cell connections. If the actual angle is larger, tangential forces act on the connected cells, "
        "aiming to reduce the angle. Conversely, if the actual angle is smaller, the tangential forces tend to enlarge this angle. With this type of force "
        "cell networks can fold back into a desired shape after deformation.";

    std::string const CellAgeTooltip = "The age of the cell in time steps.";

    std::string const CellIdTooltip = "The id of the cell is a unique 64 bit number which identifies the cell in the entire world and cannot be changed. The "
                                      "cell id is displayed here in hexadecimal notation.";

    std::string const CellMutationIdTooltip =
        "The mutation id is a value to distinguish mutants. After most mutations (except neural network and cell properties) the mutation id changes.";

    std::string const CellCreatureIdTooltip =
        "This value loosely identifies a specific creature. While not guaranteed, it is very likely that two creatures will have different creature ids.";

    std::string const CellLivingStateTooltip =
        "Cells can exist in various states. When a cell network is being constructed, its cells are in the 'Under construction' state. Once the cell network "
        "is completed by the constructor, the cells briefly enter the 'Activating' state before transitioning to the 'Ready' state shortly after. If a cell "
        "network is in the process of dying, its cells are in the 'Dying' state.";

    inline std::string getCellFunctionTooltip(CellFunction cellFunction)
    {
        switch (cellFunction) {
        case CellFunction_Neuron:
            return Const::NeuronTooltip;
        case CellFunction_Transmitter:
            return Const::TransmitterTooltip;
        case CellFunction_Constructor:
            return Const::ConstructorTooltip;
        case CellFunction_Sensor:
            return Const::SensorTooltip;
        case CellFunction_Nerve:
            return Const::NerveTooltip;
        case CellFunction_Attacker:
            return Const::AttackerTooltip;
        case CellFunction_Injector:
            return Const::InjectorTooltip;
        case CellFunction_Muscle:
            return Const::MuscleTooltip;
        case CellFunction_Defender:
            return Const::DefenderTooltip;
        case CellFunction_Reconnector:
            return Const::ReconnectorTooltip;
        case CellFunction_Detonator:
            return Const::DetonatorTooltip;
        default:
            return Const::CellFunctionTooltip;
        }
    };

    std::string const GenomeNumCellsRecursivelyTooltip = "The number of all encoded cells in the genome including its sub-genomes.";

    std::string const GenomeBytesTooltip = "The length of the genome in bytes.";

    std::string const GenomeGenerationTooltip = "This value indicates the number of times this genome has been inherited by offspring.";

    std::string const GenomeNumCellsTooltip = "The number of all encoded cells in the genome excluding its sub-genomes.";

    std::string const GenomeCurrentCellTooltip = "The sequence number of the cell in the genome that will be constructed next.";

    std::string const GenomeCurrentRepetitionTooltip = "The cell network encoded in the genome can be repeated in a single construction by specifying a number of "
                                                 "repetitions. This value indicates the index of the current repetition.";

    std::string const CellInjectorCounterTooltip =
        "When a genome injection is initiated, the counter increments after each consecutive successful activation of the injector. Once the counter reaches a "
        "specific threshold (refer to the 'Injection time' simulation parameter), the injection process is completed.";

    std::string const CellSensorTargetCreatureIdTooltip = "The id of the last creature that has been scanned.";

    std::string const NeuronInputTooltipByChannel[8] = {
        "The following cell functions write their output to channel #0:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron\n\n" ICON_FA_CHEVRON_RIGHT " Constructor: 0 (could not "
        "constructor next cell, e.g. no energy, required connection check failed, completeness check failed), 1 (next cell construction "
        "successful)\n\n" ICON_FA_CHEVRON_RIGHT " Sensor: 0 (no match) or 1 (match)\n\n" ICON_FA_CHEVRON_RIGHT " Attacker: a value which is proportional to the gained "
        "energy\n\n" ICON_FA_CHEVRON_RIGHT " Injector: 0 (no cells found) or 1 (injection in process or completed)\n\n" ICON_FA_CHEVRON_RIGHT
        " Reconnector: 0 (no connection created/removed) or 1 (connection created/removed)",
        "The following cell functions write their output to channel #1:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron\n\n" ICON_FA_CHEVRON_RIGHT " Sensor: density of the last match",
        "The following cell functions write their output to channel #2:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron\n\n" ICON_FA_CHEVRON_RIGHT " Sensor: distance of the last match (0 = far away, 1 = close)",
        "The following cell functions write their output to channel #3:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron\n\n" ICON_FA_CHEVRON_RIGHT " Sensor: angle of the last match",
        "The following cell functions write their output to channel #4:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron",
        "The following cell functions write their output to channel #5:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron",
        "The following cell functions write their output to channel #6:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron",
        "The following cell functions write their output to channel #7:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron\n\n" ICON_FA_CHEVRON_RIGHT " Attacker: 1 if a cell is attacked by an other attacker cell" 
    };

    std::string const NeuronOutputTooltipByChannel[8] = {
        "The following cell functions obtain their input from channel #0:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron\n\n" ICON_FA_CHEVRON_RIGHT " Constructor: abs(value) > "
        "threshold activates constructor (only necessary in 'Manual' mode)\n\n" ICON_FA_CHEVRON_RIGHT " Sensor: abs(value) > threshold activates "
        "sensor\n\n" ICON_FA_CHEVRON_RIGHT " Attacker: abs(value) > threshold activates attacker\n\n" ICON_FA_CHEVRON_RIGHT " Injector: abs(value) > threshold "
        "activates injector\n\n" ICON_FA_CHEVRON_RIGHT " Muscle: The strength of the movement, bending or expansion/contraction. A negative sign corresponds to "
        "the opposite action.\n\n" ICON_FA_CHEVRON_RIGHT " Reconnector: value > threshold triggers creation of a bond to a cell in the vicinity, value < -threshold triggers destruction of a bond\n\n"
        ICON_FA_CHEVRON_RIGHT " Detonator: abs(value) > threshold activates detonator",
        "The following cell functions obtain their input from channel #1:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron\n\n" ICON_FA_CHEVRON_RIGHT " Muscle: This channel is "
        "solely utilized for acceleration due to bending. If the sign of channel #1 differs from the sign of channel #0, no acceleration will be obtained "
        "during the bending process.",
        "The following cell functions obtain their input from channel #2:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron",
        "The following cell functions obtain their input from channel #3:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron\n\n" ICON_FA_CHEVRON_RIGHT
        " Muscle: This channel is used for muscles in movement mode. It allows to determine the relative angle of the movement. A value of -0.5 correspond to "
        "-180 deg and +0.5 to +180 deg.",
        "The following cell functions obtain their input from channel #4:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron",
        "The following cell functions obtain their input from channel #5:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron",
        "The following cell functions obtain their input from channel #6:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron",
        "The following cell functions obtain their input from channel #7:\n\n" ICON_FA_CHEVRON_RIGHT " Neuron"
    };

    std::string const CreatorPencilRadiusTooltip = "The radius of the pencil in number of cells.";

    std::string const CreatorAscendingExecutionOrderNumberTooltip =
        "Each generated cell has an 'execution order number' that is one greater than the previous generated cell.";

    std::string const CreatorRectangleWidthTooltip = "The width of the rectangle in cells.";

    std::string const CreatorRectangleHeightTooltip = "The height of the rectangle in cells.";

    std::string const CreatorHexagonLayersTooltip = "The number of layers in cells starting from the center.";

    std::string const CreatorDiscOuterRadiusTooltip = "The outer radius of the disc in cells.";

    std::string const CreatorDiscInnerRadiusTooltip = "The inner radius of the disc in cells.";

    std::string const CreatorDistanceTooltip = "The distance between two connected cells.";

    std::string const CreatorStickyTooltip = "If the Sticky property is selected, the created cells can usually form further connections. That is, they can "
                                             "'stick together' with other cell networks after collision.";

    std::string const LoginHowToCreateNewUseTooltip = "Please enter the desired user name and password and proceed by clicking the 'Create user' button.";

    std::string const LoginForgotYourPasswordTooltip = "Please enter the user name and proceed by clicking the 'Reset password' button.";

    std::string const LoginSecurityInformationTooltip =
        "The data transfer to the server is encrypted via https. On the server side, the password is not stored in cleartext, but as a salted SHA-256 hash "
        "value in the database. If the toggle 'Remember' is activated, the password will be stored in the Windows registry under the path 'HKEY_CURRENT_USER\\SOFTWARE\\alien' "
        "or, in the case of other OS, in 'settings.json' on your local machine.";

    std::string const LoginRememberTooltip = "If the toggle 'Remember' is activated, the password will be stored in the Windows registry under the path "
                                             "'HKEY_CURRENT_USER\\SOFTWARE\\alien' or, in the case of other OS, in 'settings.json' on your local machine. It "
                                             "is recommended not to choose a password that is used elsewhere.";

    std::string const LoginShareGpuInfoTooltip1 =
        "If this option is enabled, other users will be able to see in the browser window that you have the following graphics card: ";
    std::string const LoginShareGpuInfoTooltip2 =
        "As a result, you will be able to see the GPU information of other registered users who have shared it.";

    std::vector<std::string> const ActivationFunctions = {"Sigmoid", "Binary step", "Identity", "Absolute value", "Gaussian"};
}
