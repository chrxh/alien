#include <iostream>
#include <fstream>
#include <QFileDialog>
#include <QMessageBox>
#include <QStyledItemDelegate>

#include "qjsonmodel.h"

#include "Base/ServiceLocator.h"

#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SimulationParametersParser.h"
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

    _model = new QJsonModel;
    ui->treeView->setModel(_model);

    updateModelFromSimulationParameters();

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
    updateSimulationParametersFromModel();

	accept();
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

    updateModelFromSimulationParameters();
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
            updateModelFromSimulationParameters();
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
        updateSimulationParametersFromModel();

        if (!SerializationHelper::saveToFile(filename.toStdString(), [&]() {
                return _serializer->serializeSimulationParameters(_simulationParameters);
            })) {
			QMessageBox msgBox(QMessageBox::Critical, "Error", Const::ErrorSaveSimulationParameters);
			msgBox.exec();
		}
    }
}

void SimulationParametersDialog::updateModelFromSimulationParameters()
{
    auto json = _serializer->serializeSimulationParameters(_simulationParameters);
    _model->loadJson(QByteArray::fromStdString(json));
    ui->treeView->expandAll();
}

void SimulationParametersDialog::updateSimulationParametersFromModel()
{
    auto json = _model->json();
    auto jsonByteArray = json.toJson();
    _simulationParameters = _serializer->deserializeSimulationParameters(jsonByteArray.toStdString());
}
