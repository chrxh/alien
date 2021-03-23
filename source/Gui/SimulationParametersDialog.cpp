#include <iostream>
#include <fstream>
#include <QFileDialog>
#include <QMessageBox>
#include <QStyledItemDelegate>

#include "Base/ServiceLocator.h"

#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/SerializationHelper.h"
#include "EngineInterface/EngineInterfaceBuilderFacade.h"

#include "Settings.h"
#include "SimulationParametersDialog.h"
#include "SimulationConfig.h"
#include "ui_simulationparametersdialog.h"

namespace {
    class NoEditDelegate : public QStyledItemDelegate {
    public:
        NoEditDelegate(QObject* parent = 0)
            : QStyledItemDelegate(parent)
        {}
        virtual QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index)
            const
        {
            return 0;
        }
    };
}

SimulationParametersDialog::SimulationParametersDialog(SimulationParameters const&  parameters, Serializer* serializer, QWidget *parent)
	: QDialog(parent), ui(new Ui::SimulationParametersDialog), _simulationParameters(parameters)
	, _serializer(serializer)
{
    ui->setupUi(this);
    ui->treeWidget->expandAll();
	ui->treeWidget->setColumnWidth(0, 270);
    ui->treeWidget->setItemDelegateForColumn(0, new NoEditDelegate(this));

    updateWidgetsFromSimulationParameters ();

    //connections
    connect(ui->defaultButton, SIGNAL(clicked()), this, SLOT(defaultButtonClicked()));
    connect(ui->loadButton, SIGNAL(clicked()), this, SLOT(loadButtonClicked()));
    connect(ui->saveButton, SIGNAL(clicked()), this, SLOT(saveButtonClicked()));
	connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &SimulationParametersDialog::okClicked);
}

SimulationParametersDialog::~SimulationParametersDialog()
{
    delete ui;
}

SimulationParameters const& SimulationParametersDialog::getSimulationParameters () const
{
    return _simulationParameters;
}

void SimulationParametersDialog::okClicked()
{
    updateSimulationParametersFromWidgets();
	accept();
}

void SimulationParametersDialog::updateWidgetsFromSimulationParameters ()
{
    setItem("min token usages", 0, _simulationParameters.cellMinTokenUsages);
    setItem("token usage decay probability", 0, _simulationParameters.cellTokenUsageDecayProb);
    setItem("min distance", 0, _simulationParameters.cellMinDistance);
	setItem("max distance", 0, _simulationParameters.cellMaxDistance);
	setItem("max force", 0, _simulationParameters.cellMaxForce);
    setItem("max force decay probability", 0, _simulationParameters.cellMaxForceDecayProb);
    setItem("max bonds", 0, _simulationParameters.cellMaxBonds);
    setItem("max token", 0, _simulationParameters.cellMaxToken);
    setItem("max token branch number", 0, _simulationParameters.cellMaxTokenBranchNumber);
    setItem("min energy", 0, _simulationParameters.cellMinEnergy);
    setItem("transformation probability", 0, _simulationParameters.cellTransformationProb);
    setItem("fusion velocity", 0, _simulationParameters.cellFusionVelocity);

    setItem("strength", 0, _simulationParameters.cellFunctionWeaponStrength);
    setItem("energy cost", 0, _simulationParameters.cellFunctionWeaponEnergyCost);
    setItem("max instructions", 0, _simulationParameters.cellFunctionComputerMaxInstructions);
    setItem("memory size", 0, _simulationParameters.cellFunctionComputerCellMemorySize);
    setItem("offspring cell energy", 0, _simulationParameters.cellFunctionConstructorOffspringCellEnergy);
    setItem("offspring cell distance", 0, _simulationParameters.cellFunctionConstructorOffspringCellDistance);
	setItem("offspring token energy", 0, _simulationParameters.cellFunctionConstructorOffspringTokenEnergy);
    setItem("token data mutation probability", 0, _simulationParameters.cellFunctionConstructorTokenDataMutationProb);
    setItem("cell data mutation probability", 0, _simulationParameters.cellFunctionConstructorCellDataMutationProb);
    setItem("cell property mutation probability", 0, _simulationParameters.cellFunctionConstructorCellPropertyMutationProb);
    setItem("cell structure mutation probability", 0, _simulationParameters.cellFunctionConstructorCellStructureMutationProb);
    setItem("range", 0, _simulationParameters.cellFunctionSensorRange);
    setItem("range", 1, _simulationParameters.cellFunctionCommunicatorRange);

	setItem("memory size", 1, _simulationParameters.tokenMemorySize);
	setItem("min energy", 1, _simulationParameters.tokenMinEnergy);

    setItem("exponent", 0, _simulationParameters.radiationExponent);
    setItem("factor", 0, _simulationParameters.radiationFactor);
    setItem("probability", 0, _simulationParameters.radiationProb);
    setItem("velocity multiplier", 0, _simulationParameters.radiationVelocityMultiplier);
    setItem("velocity perturbation", 0, _simulationParameters.radiationVelocityPerturbation);
}

void SimulationParametersDialog::updateSimulationParametersFromWidgets ()
{
    _simulationParameters.cellMinTokenUsages = getItemReal("min token usages", 0);
    _simulationParameters.cellTokenUsageDecayProb = getItemReal("token usage decay probability", 0);
    _simulationParameters.cellMinDistance = getItemReal("min distance", 0);
	_simulationParameters.cellMaxDistance = getItemReal("max distance", 0);
    _simulationParameters.cellMaxForce = getItemReal("max force", 0);
    _simulationParameters.cellMaxForceDecayProb = getItemReal("max force decay probability", 0);
    _simulationParameters.cellMaxBonds = getItemInt("max bonds", 0);
    _simulationParameters.cellMaxToken = getItemInt("max token", 0);
    _simulationParameters.cellMaxTokenBranchNumber = getItemInt("max token branch number", 0);
    _simulationParameters.cellMinEnergy = getItemReal("min energy", 0);
    _simulationParameters.cellTransformationProb = getItemReal("transformation probability", 0);
    _simulationParameters.cellFusionVelocity = getItemReal("fusion velocity", 0);

    _simulationParameters.cellFunctionWeaponStrength = getItemReal("strength", 0);
    _simulationParameters.cellFunctionWeaponEnergyCost = getItemReal("energy cost", 0);
    _simulationParameters.cellFunctionComputerMaxInstructions = getItemInt("max instructions", 0);
    _simulationParameters.cellFunctionComputerCellMemorySize = getItemInt("memory size", 0);
    _simulationParameters.cellFunctionConstructorOffspringCellEnergy = getItemReal("offspring cell energy", 0);
    _simulationParameters.cellFunctionConstructorOffspringCellDistance = getItemReal("offspring cell distance", 0);
    _simulationParameters.cellFunctionConstructorOffspringTokenEnergy = getItemReal("offspring token energy", 0);
    _simulationParameters.cellFunctionConstructorTokenDataMutationProb = getItemReal("token data mutation probability", 0);
    _simulationParameters.cellFunctionConstructorCellDataMutationProb = getItemReal("cell data mutation probability", 0);
    _simulationParameters.cellFunctionConstructorCellPropertyMutationProb = getItemReal("cell property mutation probability", 0);
    _simulationParameters.cellFunctionConstructorCellStructureMutationProb = getItemReal("cell structure mutation probability", 0);
    _simulationParameters.cellFunctionSensorRange = getItemReal("range", 0);
    _simulationParameters.cellFunctionCommunicatorRange = getItemReal("range", 1);

	_simulationParameters.tokenMemorySize = getItemInt("memory size", 1);
    _simulationParameters.tokenMinEnergy = getItemReal("min energy", 1);

    _simulationParameters.radiationExponent = getItemReal("exponent", 0);
    _simulationParameters.radiationFactor = getItemReal("factor", 0);
    _simulationParameters.radiationProb = getItemReal("probability", 0);
    _simulationParameters.radiationVelocityMultiplier = getItemReal("velocity multiplier", 0);
    _simulationParameters.radiationVelocityPerturbation = getItemReal("velocity perturbation", 0);
}

void SimulationParametersDialog::setItem(QString key, int matchPos, int value)
{
	ui->treeWidget->findItems(key, Qt::MatchExactly | Qt::MatchRecursive).at(matchPos)->setText(1, QString("%1").arg(value));
}

void SimulationParametersDialog::setItem(QString key, int matchPos, qreal value)
{
	ui->treeWidget->findItems(key, Qt::MatchExactly | Qt::MatchRecursive).at(matchPos)->setText(1, QString("%1").arg(value));
}

int SimulationParametersDialog::getItemInt(QString key, int matchPos)
{
	bool ok(true);
	return ui->treeWidget->findItems(key, Qt::MatchExactly | Qt::MatchRecursive).at(matchPos)->text(1).toInt(&ok);
}

qreal SimulationParametersDialog::getItemReal(QString key, int matchPos)
{
	bool ok(true);
	return ui->treeWidget->findItems(key, Qt::MatchExactly | Qt::MatchRecursive).at(matchPos)->text(1).toDouble(&ok);
}

bool SimulationParametersDialog::saveSimulationParameters(string filename)
{
	try {
		std::ofstream stream(filename, std::ios_base::out | std::ios_base::binary);
		string const& data = _serializer->serializeSimulationParameters(_simulationParameters);;
		size_t dataSize = data.size();
		stream.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
		stream.write(&data[0], data.size());
		stream.close();
		if (stream.fail()) {
			return false;
		}
	}
	catch (...) {
		return false;
	}
	return true;
}

void SimulationParametersDialog::defaultButtonClicked ()
{
    auto EngineInterfaceFacade = ServiceLocator::getInstance().getService<EngineInterfaceBuilderFacade>();
	_simulationParameters = EngineInterfaceFacade->getDefaultSimulationParameters();
    updateWidgetsFromSimulationParameters();
}

void SimulationParametersDialog::loadButtonClicked ()
{
    QString filename = QFileDialog::getOpenFileName(this, "Load Simulation Parameters", "", "Alien Simulation Parameters(*.json)");
    if( !filename.isEmpty() ) {
		auto origSimulationParameters = _simulationParameters;
        bool success = true;
        try {
            success = SerializationHelper::loadFromFile<SimulationParameters>(
                filename.toStdString(),
                [&](string const& data) { return _serializer->deserializeSimulationParameters(data); },
                _simulationParameters);
        } catch (...) {
            success = false;
        }
		if (success) {
			updateWidgetsFromSimulationParameters();
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", Const::ErrorLoadSimulationParameters);
			msgBox.exec();
			_simulationParameters = origSimulationParameters;
		}
    }
}

void SimulationParametersDialog::saveButtonClicked ()
{
    QString filename = QFileDialog::getSaveFileName(this, "Save Simulation Parameters", "", "Alien Simulation Parameters(*.json)");
    if( !filename.isEmpty() ) {
		updateSimulationParametersFromWidgets();
		if (!SerializationHelper::saveToFile(filename.toStdString(), [&]() { return _serializer->serializeSimulationParameters(_simulationParameters); })) {
			QMessageBox msgBox(QMessageBox::Critical, "Error", Const::ErrorSaveSimulationParameters);
			msgBox.exec();
		}
    }
}
