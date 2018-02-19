#include <iostream>
#include <fstream>

#include <QTime>
#include <QCoreApplication>

#include "Base/GlobalFactory.h"
#include "Base/ServiceLocator.h"
#include "Base/NumberGenerator.h"

#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Api/SimulationController.h"
#include "Model/Api/SimulationContext.h"
#include "Model/Api/SpaceProperties.h"
#include "Model/Api/SimulationAccess.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Api/Serializer.h"
#include "Model/Api/DescriptionHelper.h"

#include "VersionController.h"
#include "SerializationHelper.h"
#include "InfoController.h"
#include "MainController.h"
#include "MainView.h"
#include "MainModel.h"
#include "DataRepository.h"
#include "Notifier.h"

MainController::MainController(QObject * parent)
	: QObject(parent)
{
}

MainController::~MainController()
{
	delete _view;
}

void MainController::init()
{
	_model = new MainModel(this);
	_view = new MainView();

	auto factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto numberGenerator = factory->buildRandomNumberGenerator();

	auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto serializer = facade->buildSerializer();
	auto simAccessForDataController = facade->buildSimulationAccess();
	auto descHelper = facade->buildDescriptionHelper();
	auto versionController = new VersionController();
	SET_CHILD(_serializer, serializer);
	SET_CHILD(_simAccess, simAccessForDataController);
	SET_CHILD(_descHelper, descHelper);
	SET_CHILD(_versionController, versionController);
	SET_CHILD(_numberGenerator, numberGenerator);
	_repository = new DataRepository(this);
	_notifier = new Notifier(this);

	connect(_serializer, &Serializer::serializationFinished, this, &MainController::serializationFinished);

	_serializer->init();
	_numberGenerator->init(12315312, 0);
	_view->init(_model, this, _serializer, _repository, _notifier, _numberGenerator);

	
	if (!onLoadSimulation("autosave.sim")) {

		//default simulation
		NewSimulationConfig config{
			8, { 12, 6 },{ 12 * 33 * 2 , 12 * 17 * 2 },
			facade->buildDefaultSymbolTable(),
			facade->buildDefaultSimulationParameters(),
			0/*20000 * 9 * 20*/
		};
		onNewSimulation(config);
	}
}

namespace
{
	void processEventsForMilliSec(int millisec)
	{
		QTime dieTime = QTime::currentTime().addMSecs(millisec);
		while (QTime::currentTime() < dieTime)
		{
			QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
		}
	}
}

void MainController::autoSave()
{
	onSaveSimulation("autosave.sim");
	processEventsForMilliSec(200);
}

void MainController::onRunSimulation(bool run)
{
	_simController->setRun(run);
	_versionController->clearStack();
}

void MainController::onStepForward()
{
	_versionController->saveSimulationContentToStack();
	_simController->calculateSingleTimestep();
}

void MainController::onStepBackward(bool& emptyStack)
{
	_versionController->loadSimulationContentFromStack();
	emptyStack = _versionController->isStackEmpty();
}

void MainController::onMakeSnapshot()
{
	_versionController->makeSnapshot();
}

void MainController::onRestoreSnapshot()
{
	_versionController->restoreSnapshot();
}

void MainController::initSimulation(SymbolTable* symbolTable, SimulationParameters const* parameters)
{
	_model->setSimulationParameters(parameters);
	_model->setSymbolTable(symbolTable);

	connectSimController();
	_simAccess->init(_simController->getContext());
	_descHelper->init(_simController->getContext());
	_versionController->init(_simController->getContext());
	_repository->init(_notifier, _simAccess, _descHelper, _simController->getContext(), _numberGenerator);

	_view->setupEditors(_simController);
}

void MainController::recreateSimulation(string const & serializedSimulation)
{
	auto origSimController = _simController;
	_simController = _serializer->deserializeSimulation(serializedSimulation);
	delete origSimController;

	auto symbolTable = _simController->getContext()->getSymbolTable();
	auto simulationParameters = _simController->getContext()->getSimulationParameters();

	initSimulation(symbolTable, simulationParameters);

	_view->refresh();
}

void MainController::onNewSimulation(NewSimulationConfig config)
{
	delete _simController;
	auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	_simController = facade->buildSimulationController(config.maxThreads, config.gridSize, config.universeSize, config.symbolTable, config.parameters);

	initSimulation(config.symbolTable, config.parameters);

	addRandomEnergy(config.energy);

	_view->refresh();
}

void MainController::onSaveSimulation(string const& filename)
{
	_jobsAfterSerialization.push_back(boost::make_shared<_SaveToFileJob>(filename));
	_serializer->serialize(_simController);
}

bool MainController::onLoadSimulation(string const & filename)
{
	auto origSimController = _simController;
	if (!SerializationHelper::loadFromFile<SimulationController*>(filename, [&](string const& data) { return _serializer->deserializeSimulation(data); }, _simController)) {
		return false;
	}
	delete origSimController;

	initSimulation(_simController->getContext()->getSymbolTable(), _simController->getContext()->getSimulationParameters());

	_view->refresh();
	return true;
}

void MainController::onRecreateSimulation(SimulationConfig const& simConfig)
{
	_jobsAfterSerialization.push_back(boost::make_shared<_RecreateJob>());
	_serializer->serialize(_simController, { simConfig.universeSize, simConfig.gridSize, simConfig.maxThreads });
}

void MainController::onUpdateSimulationParametersForRunningSimulation()
{
	_simController->getContext()->setSimulationParameters(_model->getSimulationParameters());
}

void MainController::onRestrictTPS(optional<int> const& tps)
{
	_simController->setRestrictTimestepsPreSecond(tps);
}

int MainController::getTimestep() const
{
	return _simController->getTimestep();
}

SimulationConfig MainController::getSimulationConfig() const
{
	auto context = _simController->getContext();
	return{ context->getMaxThreads(), context->getGridSize(), context->getSpaceProperties()->getSize() };
}

void MainController::connectSimController() const
{
	connect(_simController, &SimulationController::nextTimestepCalculated, [this]() {
		_view->getInfoController()->increaseTimestep();
	});
}

void MainController::addRandomEnergy(double amount)
{
	double maxEnergyPerCell = _simController->getContext()->getSimulationParameters()->cellMinEnergy;
	_repository->addRandomParticles(amount, maxEnergyPerCell);
	Q_EMIT _notifier->notify({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::ActionController
	}, UpdateDescription::All);

}

void MainController::serializationFinished()
{
	for (auto job : _jobsAfterSerialization) {
		if (job->type == _AsyncJob::Type::SaveToFile) {
			auto saveToFileJob = boost::static_pointer_cast<_SaveToFileJob>(job);
			SerializationHelper::saveToFile(saveToFileJob->filename, [&]() { return _serializer->retrieveSerializedSimulation(); });
		}
		if (job->type == _AsyncJob::Type::Recreate) {
			auto recreateJob = boost::static_pointer_cast<_RecreateJob>(job);
			recreateSimulation(_serializer->retrieveSerializedSimulation());
		}
	}
	_jobsAfterSerialization.clear();
}
