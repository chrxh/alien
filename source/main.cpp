#include <QApplication>

#include "gui/mainwindow.h"
#include "model/simulationcontroller.h"
#include "model/metadatamanager.h"
#include "model/simulationsettings.h"
#include "model/entities/cellto.h"

#include <QtCore/qmath.h>
#include <QVector2D>

//QT += webkitwidgets

//Design-Entscheidung:
//- alle Serialisierung und Deserialisierung sollen von SerializationFacade gesteuert werden
//- (de)serialisierung elementarer(Qt) Typen in den Methoden (de)serialize(...)

//Nächstes Mal:
//- SimulationContext in den Implementierungen der CellFeatures verwenden

//Model-Refactoring:
//- factoryfacade in featuredcellfactory umbenennen
//- in AlienCellFunctionComputerImpl: getInternalData und getMemoryReference vereinheitlichen
//- Prefix Alien in Klassen entfernen
//- suche nach "TODO"
//- AlienSimulator::updateCell: Konzept für cell->setCellFunction(AlienCellFunctionFactory::build(newCellData.cellFunctionName, false, _grid));
//- Radiation als Feature
//- SimulationSettings und MetadataManager über Grid erreichbar
//- Grid zerteilen
//- NICHT Set<CellCluster*> in SimulationController::updateCell benutzen!!!

//Potentielle Fehlerquellen:
//- Serialisierung von int (32 oder 64 Bit)
//- AlienCellCluster: calcTransform() nach setPosition(...) aufrufen
//- ShapeUniverse: _grid->correctPosition(pos) nach Positionsänderungen aufrufen
//- ReadSimulationParameters VOR Clusters lesen

//Optimierung:
//- bei AlienCellFunctionConstructor: Energie im Vorfeld checken
//- bessere Datenstrukturen für _highlightedCells in ShapeUniverse (nur Cluster abspeichern?)

//Issues:
//- bei Pfeile zwischen Zellen im Microeditor berücksichtigen, wenn Branchnumber anders gesetzt wird
//- Anzeigen, ob schon compiliert ist

//Bugs:
//- Farben bei neuen Einträgen in SymbolTable stimmt nicht
//- Computer-Code-Editor-Bug (wenn man viele Zeilen schreibt, verschwindet der Cursor am unteren Rand)
//- verschieben überlagerter Cluster im Editor: Map wird falsch aktualisiert
//- Editormodus: nach Multiplier werden Cluster focussiert
//- Zahlen werden ständig niedriger im ClusterEditor wenn man die Pfeiltasten drückt und die Werte negativ sind
//- backspace bei metadata:cell name
//- große Cluster Verschieben=> werden nicht richtig gelöscht und neu gezeichnet...
//- Arbeiten mit dem Makro-Editor bei laufender Simulation erzeugt Fehler (weil auf cells zugegriffen werden, die vielleicht nicht mehr existieren)
//- Fehler bei der Winkelmessung in AlienCellFunctionScanner

//TODO (nächstes Mal):
//- CellFunction Sensor: nehegelegene Masse ab vorgegebener Größe orten

//TODO (kurzfristig):
//- Computer-Code-Editor: Meldung, wenn maximale Zeilen überschritten sind
//- manuelle Änderungen der Geschwindigkeit soll Cluster nicht zerreißen
//- Editor-Modus: Cell fokussieren, dann Calc Timestep: Verschiebung in der Ansicht vermeiden
//- Editor-Modus: bei Rotation: mehrere Cluster um gemeinsamen Mittelpunkt rotieren
//- Einstellungen automatisch speichern
//- Tokenbranchnumber anpassen, wenn Zelle editiert wird
//- Metadata-Coordinator static machen
//- "Bild auf" / "Bild ab" im Mikroeditor zum Wechseln zur nächsten Zelle
//- AlienCellFunctionConstructur: BUILD_CONNECTION- Modus (d.h. neue Strukturen werden mit anderen überall verbunden)

//TODO (langfristig):
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

//Optional:
//- Cell-memory im Scanner auslesen und im Constructor erstellen

int main(int argc, char *argv[])
{
    //register types
    qRegisterMetaType<CellTO>("CellTO");
//	qRegisterMetaType<SimulationContext>("SimulationContext");

    //load default metadata
//    Metadata::loadDefaultMetadata(&MetadataManager::getGlobalInstance());

    //init main objects
    QApplication a(argc, argv);
    SimulationController simulator(QVector2D(400, 200), SimulationController::Threading::EXTRA_THREAD);
    MainWindow w(&simulator);
    w.setWindowState(w.windowState() | Qt::WindowFullScreen);

    w.show();
	return a.exec();
}

