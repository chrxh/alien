#include <iostream>
#include <fstream>

#include <QTime>
#include <QCoreApplication>

#include "Base/GlobalFactory.h"
#include "Base/ServiceLocator.h"

#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/Serializer.h"
#include "ModelBasic/DescriptionHelper.h"
#include "ModelBasic/SimulationMonitor.h"
#include "ModelBasic/SerializationHelper.h"

#include "ModelCpu/SimulationControllerCpu.h"
#include "ModelCpu/SimulationAccessCpu.h"
#include "ModelCpu/ModelCpuBuilderFacade.h"
#include "ModelCpu/ModelCpuData.h"
#include "ModelCpu/SimulationMonitorCpu.h"

#include "ModelGpu/SimulationAccessGpu.h"
#include "ModelGpu/SimulationControllerGpu.h"
#include "ModelGpu/ModelGpuBuilderFacade.h"
#include "ModelGpu/ModelGpuData.h"
#include "ModelGpu/SimulationMonitorGpu.h"

#include "VersionController.h"
#include "InfoController.h"
#include "MainController.h"
#include "MainView.h"
#include "MainModel.h"
#include "DataRepository.h"
#include "Notifier.h"
#include "SimulationConfig.h"

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

	_controllerBuildFunc = [](int typeId, IntVector2D const& universeSize, SymbolTable* symbols,
		SimulationParameters const& parameters, map<string, int> const& typeSpecificData, uint timestepAtBeginning) -> SimulationController*
	{
		if (ModelComputationType(typeId) == ModelComputationType::Cpu) {
			auto facade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
			ModelCpuData data(typeSpecificData);
			return facade->buildSimulationController({ universeSize, symbols, parameters }, data, timestepAtBeginning);
		}
		else if (ModelComputationType(typeId) == ModelComputationType::Gpu) {
			auto facade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();
			ModelGpuData data(typeSpecificData);
			return facade->buildSimulationController({ universeSize, symbols, parameters }, data, timestepAtBeginning);
		}
		else {
			THROW_NOT_IMPLEMENTED();
		}
	};
	_accessBuildFunc = [](SimulationController* controller) -> SimulationAccess*
	{
		if (auto controllerCpu = dynamic_cast<SimulationControllerCpu*>(controller)) {
			auto modelCpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
			SimulationAccessCpu* access = modelCpuFacade->buildSimulationAccess();
			access->init(controllerCpu);
			return access;
		}
		else if (auto controllerGpu = dynamic_cast<SimulationControllerGpu*>(controller)) {
			auto modelGpuFacade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();
			SimulationAccessGpu* access = modelGpuFacade->buildSimulationAccess();
			access->init(controllerGpu);
			return access;
		}
		else {
			THROW_NOT_IMPLEMENTED();
		}
	};
	_monitorBuildFunc = [](SimulationController* controller) -> SimulationMonitor*
	{
		if (auto controllerCpu = dynamic_cast<SimulationControllerCpu*>(controller)) {
			auto facade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
			SimulationMonitorCpu* moni = facade->buildSimulationMonitor();
			moni->init(controllerCpu);
			return moni;
		}
		else if (auto controllerGpu = dynamic_cast<SimulationControllerGpu*>(controller)) {
			auto facade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();
			SimulationMonitorGpu* moni = facade->buildSimulationMonitor();
			moni->init(controllerGpu);
			return moni;
		}
		else {
			THROW_NOT_IMPLEMENTED();
		}
	};

	auto modelBasicFacade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();
	auto modelCpuFacade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
	auto serializer = modelBasicFacade->buildSerializer();
	auto simAccessForDataController = modelCpuFacade->buildSimulationAccess();
	auto descHelper = modelBasicFacade->buildDescriptionHelper();
	auto versionController = new VersionController();
	SET_CHILD(_serializer, serializer);
	SET_CHILD(_simAccess, simAccessForDataController);
	SET_CHILD(_descHelper, descHelper);
	SET_CHILD(_versionController, versionController);
	_repository = new DataRepository(this);
	_notifier = new Notifier(this);

	connect(_serializer, &Serializer::serializationFinished, this, &MainController::serializationFinished);

	_serializer->init(_controllerBuildFunc, _accessBuildFunc);
	_view->init(_model, this, _serializer, _repository, _simMonitor, _notifier);

	if (!onLoadSimulation("autosave.sim")) {

		//default simulation
		auto config = boost::make_shared<_SimulationConfigCpu>();
		config->maxThreads = 8;
		config->gridSize = IntVector2D({ 12, 6 });
		config->universeSize = IntVector2D({ 12 * 33 * 2 , 12 * 17 * 2 });
		config->symbolTable = modelBasicFacade->buildDefaultSymbolTable();
		config->parameters = modelBasicFacade->buildDefaultSimulationParameters();
		onNewSimulation(config, 0);
	}

	auto config = getSimulationConfig();
	if (boost::dynamic_pointer_cast<_SimulationConfigCpu>(config)) {
		_view->getInfoController()->setDevice(InfoController::Device::CPU);
	}
	else if (boost::dynamic_pointer_cast<_SimulationConfigGpu>(config)) {
		_view->getInfoController()->setDevice(InfoController::Device::GPU);
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

void MainController::initSimulation(SymbolTable* symbolTable, SimulationParameters const& parameters)
{
	_model->setSimulationParameters(parameters);
	_model->setSymbolTable(symbolTable);

	connectSimController();

	_simAccess = _accessBuildFunc(_simController);
	auto context = _simController->getContext();
	_descHelper->init(context);
	_versionController->init(_simController->getContext(), _accessBuildFunc(_simController));
	_repository->init(_notifier, _simAccess, _descHelper, context);

	auto simMonitor = _monitorBuildFunc(_simController);
	SET_CHILD(_simMonitor, simMonitor);

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

void MainController::onNewSimulation(SimulationConfig const& config, double energyAtBeginning)
{
	delete _simController;
	if (auto configCpu = boost::dynamic_pointer_cast<_SimulationConfigCpu>(config)) {
		auto facade = ServiceLocator::getInstance().getService<ModelCpuBuilderFacade>();
		ModelCpuBuilderFacade::Config simulationControllerConfig{ configCpu->universeSize, configCpu->symbolTable, configCpu->parameters };
		ModelCpuData data(configCpu->maxThreads, configCpu->gridSize);
		_simController = facade->buildSimulationController(simulationControllerConfig, data);
	}
	else if (auto configGpu = boost::dynamic_pointer_cast<_SimulationConfigGpu>(config)) {
		auto facade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();
		ModelGpuBuilderFacade::Config simulationControllerConfig{ configGpu->universeSize, configGpu->symbolTable, configGpu->parameters };
        ModelGpuData data;
        data.setNumThreadsPerBlock(16);
        data.setNumBlocks(64*8);

        data.setNumClusterPointerArrays(1);
        data.setMaxClusters(500000);
        data.setMaxCells(2000000);
        data.setMaxParticles(2000000);
        data.setMaxTokens(500000);
        data.setMaxCellPointers(2000000 * 10);
        data.setMaxClusterPointers(500000 * 10);
        data.setMaxParticlePointers(2000000 * 10);
        data.setMaxTokenPointers(500000 * 10);

		_simController = facade->buildSimulationController(simulationControllerConfig, data);
	}
	else {
		THROW_NOT_IMPLEMENTED();
	}

	initSimulation(config->symbolTable, config->parameters);

	addRandomEnergy(energyAtBeginning);

	_view->refresh();
}

void MainController::onSaveSimulation(string const& filename)
{
	_jobsAfterSerialization.push_back(boost::make_shared<_SaveToFileJob>(filename));
	if (dynamic_cast<SimulationControllerCpu*>(_simController)) {
		_serializer->serialize(_simController, int(ModelComputationType::Cpu));
	}
	else if (dynamic_cast<SimulationControllerGpu*>(_simController)) {
		_serializer->serialize(_simController, int(ModelComputationType::Gpu));
	}
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

void MainController::onRecreateSimulation(SimulationConfig const& config)
{
	_jobsAfterSerialization.push_back(boost::make_shared<_RecreateJob>());

	if (auto configCpu = boost::dynamic_pointer_cast<_SimulationConfigCpu>(config)) {
		ModelCpuData data(configCpu->maxThreads, configCpu->gridSize);
		Serializer::Settings settings{ configCpu->universeSize, data.getData() };
		_serializer->serialize(_simController, int(ModelComputationType::Cpu), settings);
	}
}

void MainController::onUpdateSimulationParametersForRunningSimulation()
{
	_simController->getContext()->setSimulationParameters(_model->getSimulationParameters());
}

void MainController::onRestrictTPS(optional<int> const& tps)
{
	_simController->setRestrictTimestepsPerSecond(tps);
}

int MainController::getTimestep() const
{
	return _simController->getTimestep();
}

SimulationConfig MainController::getSimulationConfig() const
{
	auto context = _simController->getContext();

	if (dynamic_cast<SimulationControllerCpu*>(_simController)) {
		ModelCpuData data(context->getSpecificData());
		auto result = boost::make_shared<_SimulationConfigCpu>();
		result->maxThreads = data.getMaxRunningThreads();
		result->gridSize = data.getGridSize();
		result->universeSize = context->getSpaceProperties()->getSize();
		result->symbolTable = context->getSymbolTable();
		result->parameters = context->getSimulationParameters();
		return result;
	}
	else if (dynamic_cast<SimulationControllerGpu*>(_simController)) {
		auto result = boost::make_shared<_SimulationConfigGpu>();
		result->universeSize = context->getSpaceProperties()->getSize();
		result->symbolTable = context->getSymbolTable();
		result->parameters = context->getSimulationParameters();
		return result;
	}
	else {
		THROW_NOT_IMPLEMENTED();
	}
}

SimulationMonitor * MainController::getSimulationMonitor() const
{
	return _simMonitor;
}

void MainController::connectSimController() const
{
	connect(_simController, &SimulationController::nextTimestepCalculated, [this]() {
		_view->getInfoController()->increaseTimestep();
	});
}

void MainController::addRandomEnergy(double amount)
{
	double maxEnergyPerCell = _simController->getContext()->getSimulationParameters().cellMinEnergy;
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
