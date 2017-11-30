#include "Base/ServiceLocator.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Api/SimulationController.h"

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
	restoreLastSession();
}

void MainController::restoreLastSession()
{
	auto origDataManipulator = _dataManipulator;
	auto origNotifier = _notifier;
	auto origSimController = _simController;

	auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto symbols = facade->buildDefaultSymbolTable();
	auto parameters = facade->buildDefaultSimulationParameters();
	IntVector2D size = { 12 * 33 * 3 /** 2 */ , 12 * 17 * 3 /** 2 */ };
	_simController = facade->buildSimulationController(8, { 12, 6 }, size, symbols, parameters);

	_dataManipulator = new DataManipulator(this);
	_notifier = new Notifier(this);
	auto descHelper = facade->buildDescriptionHelper(_simController->getContext());
	auto access = facade->buildSimulationAccess(_simController->getContext());
	_dataManipulator->init(_notifier, access, descHelper, _simController->getContext());

	_view->setupEditors(_simController, _dataManipulator, _notifier);

	delete origDataManipulator;
	delete origNotifier;
	delete origSimController;
}
