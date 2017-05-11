#include <QApplication>
#include <QtCore/qmath.h>
#include <QVector2D>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "gui/MainWindow.h"
#include "model/AccessPorts/SimulationAccess.h"
#include "model/BuilderFacade.h"
#include "model/Settings.h"
#include "model/SimulationController.h"
#include "model/Context/UnitContext.h"
#include "model/Context/SimulationParameters.h"
#include "model/Entities/CellTO.h"
#include "model/Metadata/SymbolTable.h"


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
//- getValueOrDefault implementieren
//- config an SimulationAccess::requireData übergeben, welche Daten geholt werden sollen

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
    qRegisterMetaType<CellTO>("CellTO");

    //init main objects
    QApplication a(argc, argv);
	SimulationController* controller;
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	IntVector2D size = { 600, 600 };
	auto metric = facade->buildSpaceMetric(size);
	auto symbols = ModelSettings::loadDefaultSymbolTable();
	auto parameters = ModelSettings::loadDefaultSimulationParameters();
	auto context = facade->buildSimulationContext(4, { 6, 6 }, metric, symbols, parameters);
	controller = facade->buildSimulationController(context);

	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto numberGen = factory->buildRandomNumberGenerator();
	numberGen->init(123123, 0);

	auto access = facade->buildSimulationAccess(context);
	DataDescription desc;
	for (int i = 0; i < 100000; ++i) {
		desc.addCellCluster(CellClusterDescription().setPos(QVector2D(numberGen->getRandomInt(size.x), numberGen->getRandomInt(size.y)))
			.setVel(QVector2D(numberGen->getRandomReal() - 0.5, numberGen->getRandomReal() - 0.5))
			.addCell(CellDescription().setEnergy(parameters->cellCreationEnergy).setMaxConnections(4)));
	}
	access->updateData(desc);

    MainWindow w(controller);
    w.setWindowState(w.windowState() | Qt::WindowFullScreen);

    w.show();
	auto result = a.exec();
	delete controller;
	return result;
}

