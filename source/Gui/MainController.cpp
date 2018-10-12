#include <iostream>
#include <fstream>

#include <QTime>
#include <QCoreApplication>

#include "Base/GlobalFactory.h"
#include "Base/ServiceLocator.h"
#include "Base/NumberGenerator.h"

#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/Serializer.h"
#include "ModelBasic/DescriptionHelper.h"
#include "ModelBasic/SimulationMonitor.h"

#include "ModelCpu/SimulationControllerCpu.h"
#include "ModelCpu/SimulationAccessCpu.h"
#include "ModelCpu/ModelCpuBuilderFacade.h"
#include "ModelCpu/ModelCpuData.h"

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

	auto modelBasicFacade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();
	auto modelCpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	auto serializer = modelBasicFacade->buildSerializer();
	auto simAccessForDataController = modelCpuFacade->buildSimulationAccess();
	auto descHelper = modelBasicFacade->buildDescriptionHelper();
	auto versionController = new VersionController();
	auto simMonitor = modelCpuFacade->buildSimulationMonitor();
	SET_CHILD(_serializer, serializer);
	SET_CHILD(_simAccess, simAccessForDataController);
	SET_CHILD(_descHelper, descHelper);
	SET_CHILD(_versionController, versionController);
	SET_CHILD(_numberGenerator, numberGenerator);
	SET_CHILD(_simMonitor, simMonitor);
	_repository = new DataRepository(this);
	_notifier = new Notifier(this);

	connect(_serializer, &Serializer::serializationFinished, this, &MainController::serializationFinished);

	_controllerBuildFunc = [](int typeId, IntVector2D const& universeSize, SymbolTable* symbols,
		SimulationParameters* parameters, map<string, int> const& typeSpecificData, uint timestepAtBeginning) -> SimulationControllerCpu*
	{
		auto modelCpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
		if (typeId == 0) {
			ModelCpuData data(typeSpecificData);
			return modelCpuFacade->buildSimulationController({ universeSize, symbols, parameters }, data, timestepAtBeginning);
		}
		return nullptr;
	};
	_accessBuildFunc = [](SimulationController* controller) -> SimulationAccess*
	{
		auto modelCpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
		SimulationAccessCpu* access = modelCpuFacade->buildSimulationAccess();
		if (auto controllerCpu = dynamic_cast<SimulationControllerCpu*>(controller)) {
			access->init(controllerCpu);
			return access;
		}
		return nullptr;
	};
	_serializer->init(_controllerBuildFunc, _accessBuildFunc);
	_numberGenerator->init(12315312, 0);
	_view->init(_model, this, _serializer, _repository, _simMonitor, _notifier, _numberGenerator);

	
	if (!onLoadSimulation("autosave.sim")) {

		//default simulation
		NewSimulationConfig config{
			8, { 12, 6 },{ 12 * 33 * 2 , 12 * 17 * 2 },
			modelBasicFacade->buildDefaultSymbolTable(),
			modelBasicFacade->buildDefaultSimulationParameters(),
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

void MainController::initSimulation(SymbolTable* symbolTable, SimulationParameters* parameters)
{
	_model->setSimulationParameters(parameters);
	_model->setSymbolTable(symbolTable);

	connectSimController();

	_simAccess = _accessBuildFunc(_simController);
	_descHelper->init(_simController->getContext());
	_versionController->init(_simController->getContext(), _accessBuildFunc(_simController));
	_repository->init(_notifier, _simAccess, _descHelper, _simController->getContext(), _numberGenerator);
	_simMonitor->init(_simController->getContext());

	SimulationAccess* accessForWidgets;
	_view->setupEditors(_simController, _accessBuildFunc(_simController));
}

void MainController::recreateSimulation(string const & serializedSimulation)
{
	delete _simController;
	_simController = _serializer->deserializeSimulation(serializedSimulation);

	auto symbolTable = _simController->getContext()->getSymbolTable();
	auto simulationParameters = _simController->getContext()->getSimulationParameters();

	initSimulation(symbolTable, simulationParameters);

	_view->refresh();
}

void MainController::onNewSimulation(NewSimulationConfig config)
{
	delete _simController;
	auto facade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	ModelCpuBuilderFacade::Config simulationControllerConfig{ config.universeSize, config.symbolTable, config.parameters };
	ModelCpuData data(config.maxThreads, config.gridSize);
	_simController = facade->buildSimulationController(simulationControllerConfig, data, 0);

	initSimulation(config.symbolTable, config.parameters);

	addRandomEnergy(config.energy);

	_view->refresh();
}

void MainController::onSaveSimulation(string const& filename)
{
	_jobsAfterSerialization.push_back(boost::make_shared<_SaveToFileJob>(filename));
	_serializer->serialize(_simController, 0);
}

bool MainController::onLoadSimulation(string const & filename)
{
	auto origSimController = _simController;	//delete later if loading failed
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
	ModelCpuData data(simConfig.maxThreads, simConfig.gridSize);
	Serializer::Settings settings{ simConfig.universeSize, data.getData() };
	_serializer->serialize(_simController, 0, settings);
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
	ModelCpuData data(context->getSpecificData());
	return{ data.getMaxRunningThreads(), data.getGridSize(), context->getSpaceProperties()->getSize() };
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
	Q_EMIT _notifier->notifyDataRepositoryChanged({
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
