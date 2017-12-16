#include <iostream>
#include <fstream>

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

#include "InfoController.h"
#include "MainController.h"
#include "MainView.h"
#include "MainModel.h"
#include "DataController.h"
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

	auto factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	_numberGenerator = factory->buildRandomNumberGenerator();
	_numberGenerator->init(12315312, 0);

	auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto serializer = facade->buildSerializer();
	auto simAccessForDataController = facade->buildSimulationAccess();
	auto descHelper = facade->buildDescriptionHelper();
	SET_CHILD(_serializer, serializer);
	SET_CHILD(_simAccess, simAccessForDataController);
	SET_CHILD(_descHelper, descHelper);
	_dataController = new DataController(this);
	_notifier = new Notifier(this);

	_serializer->init();
	connect(_serializer, &Serializer::serializationFinished, this, &MainController::serializationFinished);

	//default simulation
	NewSimulationConfig config{
		8, { 12, 6 },{ 12 * 33 * 3 , 12 * 17 * 3 },
		facade->buildDefaultSymbolTable(),
		facade->buildDefaultSimulationParameters(),
		20000 * 9
	};
	onNewSimulation(config);
}

void MainController::onRunSimulation(bool run)
{
	_simController->setRun(run);
}

void MainController::onNewSimulation(NewSimulationConfig config)
{
	delete _simController;

	_model->setSimulationParameters(config.parameters);
	_model->setSymbolTable(config.symbolTable);

	auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	_simController = facade->buildSimulationController(config.maxThreads, config.gridSize, config.universeSize, config.symbolTable, config.parameters);
	connectSimController();
	_view->getInfoController()->setTimestep(0);
	_simAccess->init(_simController->getContext());
	_descHelper->init(_simController->getContext());
	_dataController->init(_notifier, _simAccess, _descHelper, _simController->getContext());

	_view->setupEditors(_simController, _dataController, _notifier);

	addRandomEnergy(config.energy);

	_view->refresh();
}

void MainController::onSaveSimulation(string const& filename)
{
	_serializationOperations.push_back({ SerializationOperation::Type::SaveToFile, filename });
	_serializer->serialize(_simController);
}

bool MainController::onLoadSimulation(string const & filename)
{
	std::ifstream stream(filename, std::ios_base::in | std::ios_base::binary);

	int timestep;
	size_t size;
	string data;

	stream.read(reinterpret_cast<char*>(&timestep), sizeof(int));
	stream.read(reinterpret_cast<char*>(&size), sizeof(size_t));
	data.resize(size);
	stream.read(&data[0], size);
	stream.close();
	if(stream.fail()) {
		return false;
	}
	try {
		delete _simController;
		_simController = _serializer->deserializeSimulation(data);
		_simAccess->init(_simController->getContext());
		connectSimController();
		_view->getInfoController()->setTimestep(timestep);

		_model->setSimulationParameters(_simController->getContext()->getSimulationParameters());
		_model->setSymbolTable(_simController->getContext()->getSymbolTable());

		auto facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
		_descHelper->init(_simController->getContext());
		_dataController->init(_notifier, _simAccess, _descHelper, _simController->getContext());

		_view->setupEditors(_simController, _dataController, _notifier);
		_view->refresh();
	}
	catch(...) {
		return false;
	}
	return true;
}

Serializer * MainController::getSerializer() const
{
	return _serializer;
}

void MainController::connectSimController() const
{
	connect(_simController, &SimulationController::nextTimestepCalculated, [this]() {
		_view->getInfoController()->increaseTimestep();
	});
}

void MainController::addRandomEnergy(double amount)
{
	DataChangeDescription desc;
	auto universeSize = _simController->getContext()->getSpaceProperties()->getSize();
	double amountPerCell = _simController->getContext()->getSimulationParameters()->cellMinEnergy;
	for (int i = 0; i < amount; ++i) {
		desc.addNewParticle(ParticleChangeDescription().setPos(QVector2D(_numberGenerator->getRandomInt(universeSize.x), _numberGenerator->getRandomInt(universeSize.y)))
			.setVel(QVector2D(_numberGenerator->getRandomReal()*2.0 - 1.0, _numberGenerator->getRandomReal()*2.0 - 1.0))
			.setEnergy(amountPerCell));
	}
	_simAccess->updateData(desc);
}

void MainController::serializationFinished()
{
	for (SerializationOperation operation : _serializationOperations) {
		if (operation.type == SerializationOperation::Type::SaveToFile) {
			string const& data = _serializer->retrieveSerializedSimulation();
			std::ofstream stream(operation.filename, std::ios_base::out | std::ios_base::binary);
			int timestep = _view->getInfoController()->getTimestep();
			size_t dataSize = data.size();
			stream.write(reinterpret_cast<char*>(&timestep), sizeof(int));
			stream.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
			stream.write(&data[0], data.size());
			stream.close();
		}
	}
	_serializationOperations.clear();
}
