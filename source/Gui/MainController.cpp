#include <iostream>
#include <fstream>

#include <QTime>
#include <QTimer>
#include <QProgressDialog>
#include <QCoreApplication>
#include <QMessageBox>

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
#include "ModelBasic/Settings.h"

#include "ModelGpu/SimulationAccessGpu.h"
#include "ModelGpu/SimulationControllerGpu.h"
#include "ModelGpu/ModelGpuBuilderFacade.h"
#include "ModelGpu/ModelGpuData.h"
#include "ModelGpu/SimulationMonitorGpu.h"

#include "MessageHelper.h"
#include "VersionController.h"
#include "InfoController.h"
#include "MainController.h"
#include "MainView.h"
#include "MainModel.h"
#include "DataRepository.h"
#include "Notifier.h"
#include "SimulationConfig.h"
#include "DataAnalyzer.h"
#include "QApplicationHelper.h"
#include "Worker.h"

namespace Const
{
    std::string const AutoSaveFilename = "autosave.sim";
    std::string const AutoSaveForLoadingFilename = "autosave_load.sim";
}

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
        if (ModelComputationType(typeId) == ModelComputationType::Gpu) {
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
        if (auto controllerGpu = dynamic_cast<SimulationControllerGpu*>(controller)) {
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
        if (auto controllerGpu = dynamic_cast<SimulationControllerGpu*>(controller)) {
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
    auto serializer = modelBasicFacade->buildSerializer();
    auto descHelper = modelBasicFacade->buildDescriptionHelper();
    auto versionController = new VersionController();
    SET_CHILD(_serializer, serializer);
    SET_CHILD(_descHelper, descHelper);
    SET_CHILD(_versionController, versionController);
    _repository = new DataRepository(this);
    _notifier = new Notifier(this);
    _dataAnalyzer = new DataAnalyzer(this);
    auto worker = new Worker(this);
    SET_CHILD(_worker, worker);

    _serializer->init(_controllerBuildFunc, _accessBuildFunc);
    _view->init(_model, this, _serializer, _repository, _simMonitor, _notifier);
    _worker->init(_serializer);

    if (!onLoadSimulation(Const::AutoSaveFilename, LoadOption::Non)) {

        //default simulation
        auto config = boost::make_shared<_SimulationConfigGpu>();
        config->numThreadsPerBlock = 32;
        config->numBlocks = 512;
        config->maxClusters = 10000;
        config->maxCells = 1000000;
        config->maxTokens = 10000;
        config->maxParticles = 1000000;
        config->dynamicMemorySize = 100000000;
        config->universeSize = IntVector2D({ 2000 , 1000 });
        config->symbolTable = modelBasicFacade->buildDefaultSymbolTable();
        config->parameters = modelBasicFacade->buildDefaultSimulationParameters();
        onNewSimulation(config, 0);
    }

    auto config = getSimulationConfig();
    if (boost::dynamic_pointer_cast<_SimulationConfigGpu>(config)) {
        _view->getInfoController()->setDevice(InfoController::Device::Gpu);
    }
    else {
        THROW_NOT_IMPLEMENTED();
    }

    //auto save every 20 min
    _autosaveTimer = new QTimer(this);
    connect(_autosaveTimer, &QTimer::timeout, this, (void(MainController::*)())(&MainController::autoSave));
    _autosaveTimer->start(1000 * 60 * 20);
}

void MainController::autoSave()
{
    auto progress = MessageHelper::createProgressDialog("Autosaving...", _view);
    autoSaveIntern(Const::AutoSaveFilename);
    delete progress;
}

void MainController::serializeSimulationAndWaitUntilFinished()
{
    QEventLoop pause;
    bool finished = false;
    auto connection = _serializer->connect(_serializer, &Serializer::serializationFinished, [&]() {
        finished = true;
        pause.quit();
    });
    if (dynamic_cast<SimulationControllerGpu*>(_simController)) {
        _serializer->serialize(_simController, int(ModelComputationType::Gpu));
    }
    else {
        THROW_NOT_IMPLEMENTED();
    }
    while (!finished) {
        pause.exec();
    }
    QObject::disconnect(connection);
}

void MainController::autoSaveIntern(std::string const& filename)
{
    saveSimulationIntern(filename);
	QApplicationHelper::processEventsForMilliSec(1000);
}

void MainController::saveSimulationIntern(string const & filename)
{
    serializeSimulationAndWaitUntilFinished();
    SerializationHelper::saveToFile(filename, [&]() { return _serializer->retrieveSerializedSimulation(); });
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
    Q_EMIT _notifier->notifyDataRepositoryChanged({
        Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor,Receiver::ActionController
    }, UpdateDescription::All);
}

void MainController::onMakeSnapshot()
{
	_versionController->makeSnapshot();
}

void MainController::onRestoreSnapshot()
{
	_versionController->restoreSnapshot();
    Q_EMIT _notifier->notifyDataRepositoryChanged({
        Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor,Receiver::ActionController
    }, UpdateDescription::All);
}

void MainController::onToggleDisplayLink(bool toggled)
{
    _simController->setEnableCalculateFrames(toggled);
}

void MainController::initSimulation(SymbolTable* symbolTable, SimulationParameters const& parameters)
{
	_model->setSimulationParameters(parameters);
    _model->setExecutionParameters(ModelSettings::getDefaultExecutionParameters());
	_model->setSymbolTable(symbolTable);

	connectSimController();

    delete _simAccess;  //for minimal memory usage deleting old object first
    _simAccess = nullptr;
	auto simAccess = _accessBuildFunc(_simController);
    SET_CHILD(_simAccess, simAccess);
	auto context = _simController->getContext();
	_descHelper->init(context);
	_versionController->init(_simController->getContext(), _accessBuildFunc(_simController));
	_repository->init(_notifier, _simAccess, _descHelper, context);
    _dataAnalyzer->init(_accessBuildFunc(_simController), _repository, _notifier);

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
	if (auto configGpu = boost::dynamic_pointer_cast<_SimulationConfigGpu>(config)) {
		auto facade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();
		ModelGpuBuilderFacade::Config simulationControllerConfig{ configGpu->universeSize, configGpu->symbolTable, configGpu->parameters };
        ModelGpuData data;
        data.setNumBlocks(configGpu->numBlocks);
        data.setNumThreadsPerBlock(configGpu->numThreadsPerBlock);
        data.setMaxClusters(configGpu->maxClusters);
        data.setMaxCells(configGpu->maxCells);
        data.setMaxParticles(configGpu->maxParticles);
        data.setMaxTokens(configGpu->maxTokens);
        data.setMaxCellPointers(configGpu->maxCells * 10);
        data.setMaxClusterPointers(configGpu->maxClusters * 10);
        data.setMaxParticlePointers(configGpu->maxParticles * 10);
        data.setMaxTokenPointers(configGpu->maxTokens * 10);
        data.setDynamicMemorySize(configGpu->dynamicMemorySize);
        data.setStringByteSize(50000000);

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
    auto progress = MessageHelper::createProgressDialog("Saving...", _view);

    saveSimulationIntern(filename);

    QApplicationHelper::processEventsForMilliSec(1000);
    delete progress;
}

bool MainController::onLoadSimulation(string const & filename, LoadOption option)
{
    auto progress = MessageHelper::createProgressDialog("Loading...", _view);

    if (LoadOption::SaveOldSim == option) {
        autoSaveIntern(Const::AutoSaveForLoadingFilename);
    }
	delete _simController;
    _simController = nullptr;

    if (!SerializationHelper::loadFromFile<SimulationController*>(filename, [&](string const& data) { return _serializer->deserializeSimulation(data); }, _simController)) {

        //load old simulation
        if (LoadOption::SaveOldSim == option) {
            CHECK(SerializationHelper::loadFromFile<SimulationController*>(
                Const::AutoSaveForLoadingFilename,
                [&](string const& data) { return _serializer->deserializeSimulation(data); },
                _simController));
        }
        delete progress;
        return false;
	}

	initSimulation(_simController->getContext()->getSymbolTable(), _simController->getContext()->getSimulationParameters());
	_view->refresh();

    delete progress;
    return true;
}

void MainController::onRecreateUniverse(SimulationConfig const& config, bool extrapolateContent)
{
    auto const recreateFunction = [&](Serializer* serializer) {
        recreateSimulation(serializer->retrieveSerializedSimulation());
    };
    _worker->addJob(boost::make_shared<_Job>(recreateFunction));

    if (auto const configGpu = boost::dynamic_pointer_cast<_SimulationConfigGpu>(config)) {
        ModelGpuData data;
        data.setNumBlocks(configGpu->numBlocks);
        data.setNumThreadsPerBlock(configGpu->numThreadsPerBlock);
        data.setMaxClusters(configGpu->maxClusters);
        data.setMaxCells(configGpu->maxCells);
        data.setMaxParticles(configGpu->maxParticles);
        data.setMaxTokens(configGpu->maxTokens);
        data.setMaxClusterPointers(configGpu->maxClusters * 10);
        data.setMaxCellPointers(configGpu->maxCells * 10);
        data.setMaxParticlePointers(configGpu->maxParticles * 10);
        data.setMaxTokenPointers(configGpu->maxTokens*10);
        data.setDynamicMemorySize(configGpu->dynamicMemorySize);
        data.setStringByteSize(1000000);

        Serializer::Settings settings{ configGpu->universeSize, data.getData(), extrapolateContent };
        _serializer->serialize(_simController, static_cast<int>(ModelComputationType::Gpu), settings);
    }
    else {
        THROW_NOT_IMPLEMENTED();
    }
}

void MainController::onUpdateSimulationParameters(SimulationParameters const& parameters)
{
    auto progress = MessageHelper::createProgressDialog("Updating simulation parameters...", _view);

	_simController->getContext()->setSimulationParameters(parameters);

    QApplicationHelper::processEventsForMilliSec(500);
    delete progress;
}

void MainController::onUpdateExecutionParameters(ExecutionParameters const & parameters)
{
    auto progress = MessageHelper::createProgressDialog("Updating execution parameters...", _view);

    _simController->getContext()->setExecutionParameters(parameters);

    QApplicationHelper::processEventsForMilliSec(500);
    delete progress;
}

void MainController::onRestrictTPS(optional<int> const& tps)
{
	_simController->setRestrictTimestepsPerSecond(tps);
}

void MainController::onAddMostFrequentClusterToSimulation()
{
    _dataAnalyzer->addMostFrequenceClusterRepresentantToSimulation();
}

int MainController::getTimestep() const
{
    if (_simController) {
        return _simController->getContext()->getTimestep();
    }
    return 0;
}

SimulationConfig MainController::getSimulationConfig() const
{
	auto context = _simController->getContext();

	if (dynamic_cast<SimulationControllerGpu*>(_simController)) {
        ModelGpuData data(context->getSpecificData());
        auto result = boost::make_shared<_SimulationConfigGpu>();
        result->numBlocks = data.getNumBlocks();
        result->numThreadsPerBlock = data.getNumThreadsPerBlock();
        result->maxClusters = data.getMaxClusters();
        result->maxCells = data.getMaxCells();
        result->maxTokens = data.getMaxTokens();
        result->maxParticles = data.getMaxParticles();
        result->dynamicMemorySize = data.getDynamicMemorySize();

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

