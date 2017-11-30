#include "Base/GlobalFactory.h"
#include "Base/ServiceLocator.h"
#include "Base/NumberGenerator.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Api/SimulationController.h"
#include "Model/Api/SimulationAccess.h"

#include "MainController.h"
#include "MainView.h"
#include "MainModel.h"
#include "DataManipulator.h"
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

	_view->init(_model, this);

	auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	NewSimulationConfig config {
		8, { 12, 6 },{ 12 * 33 * 3 /** 2 */ , 12 * 17 * 3 /** 2 */ },
		facade->buildDefaultSymbolTable(),
		facade->buildDefaultSimulationParameters()
	};
	newSimulation(config);

	//temp
	auto factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto numberGen = factory->buildRandomNumberGenerator();
	numberGen->init(12315312, 0);
	auto access = facade->buildSimulationAccess(_simController->getContext());
	DataChangeDescription desc;
	for (int i = 0; i < 20000 * 9 /**4 */ ; ++i) {
		desc.addNewParticle(ParticleChangeDescription().setPos(QVector2D(numberGen->getRandomInt(config.universeSize.x), numberGen->getRandomInt(config.universeSize.y)))
			.setVel(QVector2D(numberGen->getRandomReal()*2.0 - 1.0, numberGen->getRandomReal()*2.0 - 1.0))
			.setEnergy(50));
	}
	_simAccess->updateData(desc);
	_view->refresh();
}

void MainController::onRunSimulation(bool run)
{
	_simController->setRun(run);
}

void MainController::newSimulation(NewSimulationConfig config)
{
	auto origDataManipulator = _dataManipulator;
	auto origNotifier = _notifier;
	auto origSimController = _simController;

	auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	_simController = facade->buildSimulationController(config.maxThreads, config.gridSize, config.universeSize, config.symbolTable, config.parameters);

	_dataManipulator = new DataManipulator(this);
	_notifier = new Notifier(this);
	auto descHelper = facade->buildDescriptionHelper(_simController->getContext());
	_simAccess = facade->buildSimulationAccess(_simController->getContext());
	_dataManipulator->init(_notifier, _simAccess, descHelper, _simController->getContext());

	_view->setupEditors(_simController, _dataManipulator, _notifier);

	delete origDataManipulator;
	delete origNotifier;
	delete origSimController;
}
