#include <iostream>
#include <fstream>
#include <sstream>

#include <QTime>
#include <QTimer>
#include <QProgressDialog>
#include <QCoreApplication>
#include <QMessageBox>
#include <QFile>

#include "Base/GlobalFactory.h"
#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"

#include "EngineInterface/EngineInterfaceBuilderFacade.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/SpaceProperties.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationMonitor.h"
#include "EngineInterface/SerializationHelper.h"
#include "EngineInterface/SimulationChanger.h"

#include "EngineGpu/SimulationAccessGpu.h"
#include "EngineGpu/SimulationControllerGpu.h"
#include "EngineGpu/EngineGpuBuilderFacade.h"
#include "EngineGpu/EngineGpuData.h"
#include "EngineGpu/SimulationMonitorGpu.h"

#include "Web/WebAccess.h"
#include "Web/WebBuilderFacade.h"

#include "SnapshotController.h"
#include "GeneralInfoController.h"
#include "MainController.h"
#include "MainView.h"
#include "MainModel.h"
#include "DataRepository.h"
#include "Notifier.h"
#include "SimulationConfig.h"
#include "DataAnalyzer.h"
#include "QApplicationHelper.h"
#include "Queue.h"
#include "WebSimulationController.h"
#include "Settings.h"
#include "MonitorController.h"
#include "StartupController.h"
#include "ProgressBar.h"

namespace Const
{
    std::string const AutoSaveFilename = "autosave.sim";
    std::string const AutoSaveForLoadingFilename = "autosave_load.sim";
}

MainController::MainController(QObject * parent)
	: QObject(parent)
{
    _view = new MainView();
}

MainController::~MainController()
{
    delete _view;
    delete _model;
}

void MainController::init()
{
    _model = new MainModel(this);

    logStart();

    _controllerBuildFunc = [](int typeId, IntVector2D const& universeSize, SymbolTable* symbols,
        SimulationParameters const& parameters, map<string, int> const& typeSpecificData, uint timestepAtBeginning) -> SimulationController*
    {
        if (ModelComputationType(typeId) == ModelComputationType::Gpu) {
            auto facade = ServiceLocator::getInstance().getService<EngineGpuBuilderFacade>();
            EngineGpuData data(typeSpecificData);
            return facade->buildSimulationController({ universeSize, symbols, parameters }, data, timestepAtBeginning);
        }
        else {
            THROW_NOT_IMPLEMENTED();
        }
    };
    _accessBuildFunc = [](SimulationController* controller) -> SimulationAccess*
    {
        if (auto controllerGpu = dynamic_cast<SimulationControllerGpu*>(controller)) {
            auto EngineGpuFacade = ServiceLocator::getInstance().getService<EngineGpuBuilderFacade>();
            SimulationAccessGpu* access = EngineGpuFacade->buildSimulationAccess();
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
            auto facade = ServiceLocator::getInstance().getService<EngineGpuBuilderFacade>();
            SimulationMonitorGpu* moni = facade->buildSimulationMonitor();
            moni->init(controllerGpu);
            return moni;
        }
        else {
            THROW_NOT_IMPLEMENTED();
        }
    };

    auto EngineInterfaceFacade = ServiceLocator::getInstance().getService<EngineInterfaceBuilderFacade>();
    auto serializer = EngineInterfaceFacade->buildSerializer();
    auto descHelper = EngineInterfaceFacade->buildDescriptionHelper();
    auto snapshotController = new SnapshotController();
    SET_CHILD(_serializer, serializer);
    SET_CHILD(_descHelper, descHelper);
    SET_CHILD(_snapshotController, snapshotController);
    _repository = new DataRepository(this);
    _notifier = new Notifier(this);
    _dataAnalyzer = new DataAnalyzer(this);
    auto worker = new Queue(this);
    SET_CHILD(_worker, worker);

    auto webFacade = ServiceLocator::getInstance().getService<WebBuilderFacade>();
    auto webAccess = webFacade->buildWebAccess();
    SET_CHILD(_webAccess, webAccess);

    auto webSimController = new WebSimulationController(webAccess, _view);
    SET_CHILD(_webSimController, webSimController);

    auto startupController = new StartupController(_webAccess, _view);

    _serializer->init(_controllerBuildFunc, _accessBuildFunc);
    _view->init(_model, this, _serializer, _repository, _notifier, _webSimController, startupController);
    _worker->init(_serializer);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    std::stringstream stream;
    stream << "loading simulation '" << Const::AutoSaveFilename << "'";
    loggingService->logMessage(Priority::Important, stream.str());

    if (!onLoadSimulation(getPathToApp() + Const::AutoSaveFilename, LoadOption::Non)) {

        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        loggingService->logMessage(Priority::Important, "simulation could not be loaded");
        loggingService->logMessage(Priority::Important, "creating new simulation instead");

        //default simulation
        auto const EngineGpuFacade = ServiceLocator::getInstance().getService<EngineGpuBuilderFacade>();

        auto config = boost::make_shared<_SimulationConfig>();
        config->cudaConstants = EngineGpuFacade->getDefaultCudaConstants();
        config->universeSize = IntVector2D({ 2000 , 1000 });
        config->symbolTable = EngineInterfaceFacade->getDefaultSymbolTable();
        config->parameters = EngineInterfaceFacade->getDefaultSimulationParameters();
        onNewSimulation(config, 0);
    }

    _view->initGettingStartedWindow();

    //auto save every 20 min
    _autosaveTimer = new QTimer(this);
    connect(_autosaveTimer, &QTimer::timeout, this, (void(MainController::*)())(&MainController::autoSave));
    _autosaveTimer->start(1000 * 60 * 20);
}

void MainController::autoSave()
{
    _progressBar = new ProgressBar("Autosaving ...", _view->getSimulationViewWidget());
    autoSaveIntern(getPathToApp() + Const::AutoSaveFilename);
    delete _progressBar;
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
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "auto saving");
    
    saveSimulationIntern(filename);
	QApplicationHelper::processEventsForMilliSec(1000);
}

void MainController::saveSimulationIntern(string const & filename)
{
    serializeSimulationAndWaitUntilFinished();
    SerializationHelper::saveToFile(filename, [&]() { return _serializer->retrieveSerializedSimulation(); });
}

string MainController::getPathToApp() const
{
    auto result = qApp->applicationDirPath();
    if (!result.endsWith("/") && !result.endsWith("\\")) {
        result += "/";
    }
    return result.toStdString();
}

void MainController::onRunSimulation(bool run)
{
	_simController->setRun(run);
	_snapshotController->clearStack();
}

void MainController::onStepForward()
{
	_snapshotController->saveSimulationContentToStack();
	_simController->calculateSingleTimestep();
}

void MainController::onStepBackward(bool& emptyStack)
{
	_snapshotController->loadSimulationContentFromStack();
	emptyStack = _snapshotController->isStackEmpty();
}

void MainController::onMakeSnapshot()
{
	_snapshotController->makeSnapshot();
}

void MainController::onRestoreSnapshot()
{
	_snapshotController->restoreSnapshot();
}

void MainController::onDisplayLink(bool toggled)
{
    _simController->setEnableCalculateFrames(toggled);
}

void MainController::onSimulationChanger(bool toggled)
{
    if (toggled) {
        auto parameters = _simController->getContext()->getSimulationParameters();
        _simChanger->activate(parameters);
    }
    else {
        _simChanger->deactivate();
    }
}

void MainController::logStart()
{
    QFile file("://Version.txt");
    CHECK(file.open(QIODevice::ReadOnly));

    QTextStream in(&file);
    auto version = in.readLine();

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, QString("initializing artificial life environment version %1").arg(version).toStdString());
}

void MainController::initSimulation(SymbolTable* symbolTable, SimulationParameters const& parameters)
{
    auto const EngineInterfaceFacade = ServiceLocator::getInstance().getService<EngineInterfaceBuilderFacade>();

	_model->setSimulationParameters(parameters);
    _model->setExecutionParameters(EngineInterfaceFacade->getDefaultExecutionParameters());
	_model->setSymbolTable(symbolTable);

	connectSimController();

	auto context = _simController->getContext();
	_descHelper->init(context);
	_snapshotController->init(_simController->getContext(), _accessBuildFunc(_simController));
	_repository->init(_notifier, _accessBuildFunc(_simController), _descHelper, context);
    _dataAnalyzer->init(_accessBuildFunc(_simController), _repository, _notifier, _serializer);

	auto simMonitor = _monitorBuildFunc(_simController);
	SET_CHILD(_simMonitor, simMonitor);

    auto webSimMonitor = _monitorBuildFunc(_simController);
    auto space = context->getSpaceProperties();
    _webSimController->init(_accessBuildFunc(_simController), webSimMonitor, getSimulationConfig());

    auto simChanger = EngineInterfaceFacade->buildSimulationChanger(simMonitor, context->getNumberGenerator());
    for (auto const& connection : _simChangerConnections) {
        QObject::disconnect(connection);
    }
    SET_CHILD(_simChanger, simChanger);
    _simChangerConnections.emplace_back(connect(
        _simController,
        &SimulationController::nextTimestepCalculated,
        _simChanger,
        &SimulationChanger::notifyNextTimestep));
    _simChangerConnections.emplace_back(connect(_simChanger, &SimulationChanger::simulationParametersChanged, [&] {
        auto newParameters = _simChanger->retrieveSimulationParameters();
        onUpdateSimulationParameters(newParameters);
    }));

	_view->initSimulation(_simController, _accessBuildFunc(_simController));
}

void MainController::recreateSimulation(SerializedSimulation const & serializedSimulation)
{
    auto ptr = _simController;
    _simController = nullptr;
    delete ptr;

	_simController = _serializer->deserializeSimulation(serializedSimulation);

	auto symbolTable = _simController->getContext()->getSymbolTable();
	auto simulationParameters = _simController->getContext()->getSimulationParameters();

    initSimulation(symbolTable, simulationParameters);

    _view->getMonitorController()->continueTimer();
	_view->refresh();
}

void MainController::onNewSimulation(SimulationConfig const& config, double energyAtBeginning)
{
    _view->getMonitorController()->pauseTimer();

    auto ptr = _simController;
    _simController = nullptr;
    delete ptr;

    auto facade = ServiceLocator::getInstance().getService<EngineGpuBuilderFacade>();
    auto simulationControllerConfig =
        EngineGpuBuilderFacade::Config{ config->universeSize, config->symbolTable, config->parameters};
    auto data = EngineGpuData(config->cudaConstants);
	_simController = facade->buildSimulationController(simulationControllerConfig, data);

	initSimulation(config->symbolTable, config->parameters);
    _view->getMonitorController()->continueTimer();

	addRandomEnergy(energyAtBeginning);

	_view->refresh();
}

void MainController::onSaveSimulation(string const& filename)
{
    _progressBar = new ProgressBar("Saving ...", _view->getSimulationViewWidget());

    saveSimulationIntern(filename);

    delete _progressBar;
}

bool MainController::onLoadSimulation(string const & filename, LoadOption option)
{
    _view->getMonitorController()->pauseTimer();

    _progressBar = new ProgressBar("Loading ...", _view->getSimulationViewWidget());

    if (LoadOption::SaveOldSim == option) {
        autoSaveIntern(getPathToApp() + Const::AutoSaveForLoadingFilename);
    }
    auto ptr = _simController;
    _simController = nullptr;
    delete ptr;

    if (!SerializationHelper::loadFromFile(
            filename,
            [&](SerializedSimulation const& data) { return _serializer->deserializeSimulation(data); },
            _simController)) {

        //load old simulation
        if (LoadOption::SaveOldSim == option) {
            CHECK(SerializationHelper::loadFromFile(
                getPathToApp() + Const::AutoSaveForLoadingFilename,
                [&](SerializedSimulation const& data) { return _serializer->deserializeSimulation(data); },
                _simController));
        }
        delete _progressBar;
        return false;
    }

	initSimulation(_simController->getContext()->getSymbolTable(), _simController->getContext()->getSimulationParameters());
    _view->getMonitorController()->continueTimer();
	_view->refresh();

    delete _progressBar;
    return true;
}

void MainController::onRecreateUniverse(SimulationConfig const& config, bool extrapolateContent)
{
    _progressBar = new ProgressBar("Reassembling world...", _view->getSimulationViewWidget());

    _view->getMonitorController()->pauseTimer();

    auto const recreateFunction = [&](Serializer* serializer) {
        recreateSimulation(serializer->retrieveSerializedSimulation());
        delete _progressBar;
        _progressBar = nullptr;
    };
    _worker->add(boost::make_shared<_ExecuteLaterFunc>(recreateFunction));

    auto data = EngineGpuData(config->cudaConstants);

    Serializer::Settings settings{ config->universeSize, data.getData(), extrapolateContent };
    _serializer->serialize(_simController, static_cast<int>(ModelComputationType::Gpu), settings);
}

void MainController::onUpdateSimulationParameters(SimulationParameters const& parameters)
{
    _progressBar = new ProgressBar("Updating simulation parameters ...", _view->getSimulationViewWidget());
    
	_simController->getContext()->setSimulationParameters(parameters);

    delete _progressBar;
}

void MainController::onUpdateExecutionParameters(ExecutionParameters const & parameters)
{
    _progressBar = new ProgressBar("Updating execution parameters ...", _view->getSimulationViewWidget());

    _simController->getContext()->setExecutionParameters(parameters);

    delete _progressBar;
}

void MainController::onRestrictTPS(boost::optional<int> const& tps)
{
    _simController->setRestrictTimestepsPerSecond(tps);
}

void MainController::onSaveRepetitiveActiveClusterAnalysis(std::string const& folder)
{
    _dataAnalyzer->saveRepetitiveActiveClustersToFiles(folder);
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
    if (!_simController) {
        return nullptr;
    }
	auto context = _simController->getContext();

	if (dynamic_cast<SimulationControllerGpu*>(_simController)) {
        auto data = EngineGpuData(context->getSpecificData());
        auto result = boost::make_shared<_SimulationConfig>();
        result->cudaConstants = data.getCudaConstants();
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
	double maxEnergyPerCell = _simController->getContext()->getSimulationParameters().cellMinEnergy * 10;
	_repository->addRandomParticles(amount, maxEnergyPerCell);
	Q_EMIT _notifier->notifyDataRepositoryChanged({
		Receiver::DataEditor,
		Receiver::Simulation,
		Receiver::VisualEditor,
		Receiver::ActionController
	}, UpdateDescription::All);
}

