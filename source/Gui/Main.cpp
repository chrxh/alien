#include <QApplication>
#include <QtCore/qmath.h>
#include <QVector2D>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/ModelBasicServices.h"
#include "ModelCpu/ModelCpuServices.h"

#include "Gui/MainController.h"


/*
#include "ModelGpu/ModelGpuBuilderFacade.h"
#include "ModelGpu/ModelGpuServices.h"
*/

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QCoreApplication::setOrganizationName("alien");
	QCoreApplication::setApplicationName("alien");

	/*
	ModelServices modelServices;
	ModelGpuServices modelGpuServices;
	ModelBuilderFacade* cpuFacade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto symbols = cpuFacade->buildDefaultSymbolTable();
	auto parameters = cpuFacade->buildDefaultSimulationParameters();
	IntVector2D size = { 12 * 33 * 3 * 3, 12 * 17 * 3 * 3 };
	ModelGpuBuilderFacade* gpuFacade = ServiceLocator::getInstance().getService<ModelGpuBuilderFacade>();
	auto controller = gpuFacade->buildSimulationController(size, symbols, parameters);
	auto access = gpuFacade->buildSimulationAccess(controller->getContext());
*/

	ModelBasicServices modelBasicServices;
	ModelCpuServices modelCpuServices;
/*
	ModelBuilderFacade* cpuFacade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto symbols = cpuFacade->buildDefaultSymbolTable();
	auto parameters = cpuFacade->buildDefaultSimulationParameters();
	IntVector2D size = { 12 * 33 * 3 / ** 2* /, 12 * 17 * 3 / ** 2* / };
	auto controller = cpuFacade->buildSimulationController(8, { 12, 6 }, size, symbols, parameters);
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto numberGen = factory->buildRandomNumberGenerator();
	numberGen->init(12315312, 0);
	auto access = cpuFacade->buildSimulationAccess(controller->getContext());
	DataChangeDescription desc;
	for (int i = 0; i < 20000*9/ **4* /; ++i) {
		desc.addNewParticle(ParticleChangeDescription().setPos(QVector2D(numberGen->getRandomInt(size.x), numberGen->getRandomInt(size.y)))
			.setVel(QVector2D(numberGen->getRandomReal()*2.0 - 1.0, numberGen->getRandomReal()*2.0 - 1.0))
			.setEnergy(50));
	}
	access->updateData(desc);

	MainView w(controller, access);
	w.setWindowState(w.windowState() | Qt::WindowFullScreen);

	w.show();
*/
	MainController controller;
	controller.init();
	return a.exec();
}

