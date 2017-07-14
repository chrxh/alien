#include <QApplication>
#include <QtCore/qmath.h>
#include <QVector2D>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "gui/MainWindow.h"
#include "Model/AccessPorts/SimulationAccess.h"
#include "Model/ModelBuilderFacade.h"
#include "Model/Settings.h"
#include "Model/SimulationController.h"
#include "Model/Context/UnitContext.h"
#include "Model/Context/SimulationParameters.h"
#include "Model/Entities/CellTO.h"
#include "Model/Metadata/SymbolTable.h"
#include "Model/ModelServices.h"

/*
#include "ModelGpu/ModelGpuBuilderFacade.h"
#include "ModelGpu/ModelGpuServices.h"
*/


//Design-Entscheidung:
//- alle Serialisierung und Deserialisierung sollen von SerializationFacade gesteuert werden
//- (de)serialisierung elementarer(Qt) Typen in den Methoden (de)serialize(...)
//- Objekte immer Default-Konstruierbar
//- Daten mit init()-Methode initialisieren
//- Factory erstellen Default-konstruierte Objekte
//- Fassaden erstellen initialisierte und einsatzbereite Objekte
//- bei QObject-Parameter in init(...) können parents gesetzt werden
//- parents werden nicht in Fassaden oder Factories übergeben
//- jeglicher Zugriff auf die Simulation (z.B. Serialisierung, Gui, ...) erfolgt über SimulationAccessApi
//- Zugriff ist asynchron
//- Zugriff verwendet Desriptions

//Nächstes Mal:
//- Prüfung Protection-Counter für cell und mapCell, setzen des Counters erst für newCell
//- Optimierung: CellMapImpl::locateCell und CellMapImpl::removeCellIfPresent
//- SimulationContext in BuilderFacadeImpl::buildSimulationController erstellen

//Model-Refactoring:
//- Serialisierungs-Framework benutzt Descriptions
//- init bei Cell/Cluster/EnergyParticle
//- C-Arrays durch Vector ersetzen (z.B. CellImpl, CellMap,...)

/**************** alte Notizen ******************/
//Potentielle Fehlerquellen Alt:
//- Serialisierung von int (32 oder 64 Bit)
//- AlienCellCluster: calcTransform() nach setPosition(...) aufrufen
//- ShapeUniverse: _grid->correctPosition(pos) nach Positionsänderungen aufrufen
//- ReadSimulationParameters VOR Clusters lesen

//Optimierung Alt:
//- bei AlienCellFunctionConstructor: Energie im Vorfeld checken
//- bessere Datenstrukturen für _highlightedCells in ShapeUniverse (nur Cluster abspeichern?)

//Issues Alt:
//- bei Pfeile zwischen Zellen im Microeditor berücksichtigen, wenn Branchnumber anders gesetzt wird
//- Anzeigen, ob schon compiliert ist
//- Cell-memory im Scanner auslesen und im Constructor erstellen

//Bugs Alt:
//- Farben bei neuen Einträgen in SymbolTable stimmt nicht
//- Computer-Code-Editor-Bug (wenn man viele Zeilen schreibt, verschwindet der Cursor am unteren Rand)
//- verschieben überlagerter Cluster im Editor: Map wird falsch aktualisiert
//- Editormodus: nach Multiplier werden Cluster focussiert
//- Zahlen werden ständig niedriger im ClusterEditor wenn man die Pfeiltasten drückt und die Werte negativ sind
//- backspace bei metadata:cell name
//- große Cluster Verschieben=> werden nicht richtig gelöscht und neu gezeichnet...
//- Arbeiten mit dem Makro-Editor bei laufender Simulation erzeugt Fehler (weil auf cells zugegriffen werden, die vielleicht nicht mehr existieren)
//- Fehler bei der Winkelmessung in AlienCellFunctionScanner

//Refactoring Alt:
//	- in AlienCellFunctionComputerImpl: getInternalData und getMemoryReference vereinheitlichen
//	- Radiation als Feature

//Todo Alt (kurzfristig):
//- Computer-Code-Editor: Meldung, wenn maximale Zeilen überschritten sind
//- manuelle Änderungen der Geschwindigkeit soll Cluster nicht zerreißen
//- Editor-Modus: Cell fokussieren, dann Calc Timestep: Verschiebung in der Ansicht vermeiden
//- Editor-Modus: bei Rotation: mehrere Cluster um gemeinsamen Mittelpunkt rotieren
//- Einstellungen automatisch speichern
//- Tokenbranchnumber anpassen, wenn Zelle editiert wird
//- Metadata-Coordinator static machen
//- "Bild auf" / "Bild ab" im Mikroeditor zum Wechseln zur nächsten Zelle
//- AlienCellFunctionConstructur: BUILD_CONNECTION- Modus (d.h. neue Strukturen werden mit anderen überall verbunden)

//Todo Alt (langfristig):
//- Color-Management
//- Geschwindigkeitsoptimierung bei vielen Zellen im Editor
//- abstoßende Kraft implementieren
//- Zellenergie im Scanner auslesen
//- Sensoreinheit implementieren
//- Multi-Thread fähig
//- Verlet-Algorithmus für Gravitationsberechnung?
//- Schädigung bei Antriebsfeuer (-teilchen)
//- bei starker Impulsänderung: Zerstoerung von langen "Korridoren" in der Struktur
//- AlienSimulator::updateCluster: am Ende nur im Grid eintragen wenn Wert 0 ?
//- in AlienGrid: setCell, removeEnergy, ... pos über AlienCell* ... auslesen

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

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

	ModelServices modelServices;
	ModelBuilderFacade* cpuFacade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto symbols = cpuFacade->buildDefaultSymbolTable();
	auto parameters = cpuFacade->buildDefaultSimulationParameters();
	IntVector2D size = { 12 * 33 * 3, 12 * 17 * 3 };
	auto controller = cpuFacade->buildSimulationController(8, { 12, 6 }, size, symbols, parameters);
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto numberGen = factory->buildRandomNumberGenerator();
	numberGen->init(123123, 0);
	auto access = cpuFacade->buildSimulationAccess(controller->getContext());
	DataDescription desc;
	for (int i = 0; i < 20000*9; ++i) {
		desc.addEnergyParticle(EnergyParticleDescription().setPos(QVector2D(numberGen->getRandomInt(size.x), numberGen->getRandomInt(size.y)))
			.setVel(QVector2D(numberGen->getRandomReal()*2.0 - 1.0, numberGen->getRandomReal()*2.0 - 1.0))
			.setEnergy(50));
	}
	access->updateData(desc);


    MainWindow w(controller, access);
    w.setWindowState(w.windowState() | Qt::WindowFullScreen);

    w.show();
	auto result = a.exec();
	delete access;
	delete controller;
	return result;
}

