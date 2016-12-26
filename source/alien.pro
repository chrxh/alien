# -------------------------------------------------
# Project created by QtCreator 2012-04-08T22:05:32
# -------------------------------------------------
QT += opengl
QMAKE_CXXFLAGS_RELEASE    = -O4 -march=native -ffast-math -funroll-loops -std=c++11 -Wno-unused-variable -Wno-unused-parameter
TARGET = alien
TEMPLATE = app
SOURCES += main.cpp \
    model/physics/physics.cpp \
    global/global.cpp \
    gui/monitoring/simulationmonitor.cpp \
    gui/microeditor/hexedit.cpp \
    gui/macroeditor/pixeluniverse.cpp \
    gui/macroeditor/shapeuniverse.cpp \
    gui/macroeditor/aliencellgraphicsitem.cpp \
    gui/macroeditor/aliencellconnectiongraphicsitem.cpp \
    gui/macroeditor/alienenergygraphicsitem.cpp \
    gui/microeditor/computercodeedit.cpp \
    gui/macroeditor.cpp \
    gui/mainwindow.cpp \
    gui/microeditor.cpp \
    gui/microeditor/clusteredit.cpp \
    gui/microeditor/tokenedit.cpp \
    gui/macroeditor/markergraphicsitem.cpp \
    gui/microeditor/celledit.cpp \
    gui/microeditor/energyedit.cpp \
    gui/microeditor/tokentab.cpp \
    gui/microeditor/symboledit.cpp \
    gui/microeditor/metadatapropertiesedit.cpp \
    gui/microeditor/metadataedit.cpp \
    gui/dialogs/newsimulationdialog.cpp \
    gui/dialogs/simulationparametersdialog.cpp \
    gui/dialogs/addenergydialog.cpp \
    gui/dialogs/symboltabledialog.cpp \
    gui/dialogs/selectionmultiplyrandomdialog.cpp \
    gui/dialogs/selectionmultiplyarrangementdialog.cpp \
    gui/microeditor/cellcomputeredit.cpp \
    gui/dialogs/addrectstructuredialog.cpp \
    gui/dialogs/addhexagonstructuredialog.cpp \
    gui/assistance/tutorialwindow.cpp \
    model/simulationsettings.cpp \
    model/metadatamanager.cpp \
    gui/misc/startscreencontroller.cpp \
    gui/guisettings.cpp \
    global/servicelocator.cpp \
    model/entities/_impl/entityfactoryimpl.cpp \
    model/entities/_impl/cellclusterimpl.cpp \
    model/entities/token.cpp \
    model/entities/grid.cpp \
    model/entities/cellto.cpp \
    model/entities/_impl/cellimpl.cpp \
    model/features/cellfeature.cpp \
    model/features/cellfunction.cpp \
    model/features/_impl/cellfeaturefactoryimpl.cpp \
    model/features/_impl/cellfunctioncomputerimpl.cpp \
    model/entities/energyparticle.cpp \
    model/simulationunit.cpp \
    model/simulationcontroller.cpp \
    model/_impl/factoryfacadeimpl.cpp \
    model/_impl/simulationcontextimpl.cpp \
    model/cellmap.cpp \
    model/definitions.cpp \
    model/energyparticlemap.cpp \
    model/topology.cpp \
    model/_impl/serializationfacadeimpl.cpp \
    model/features/_impl/energyguidanceimpl.cpp \
    model/features/_impl/cellfunctioncommunicatorimpl.cpp \
    model/features/_impl/cellfunctionconstructorimpl.cpp \
    model/features/_impl/cellfunctionpropulsionimpl.cpp \
    model/features/_impl/cellfunctionscannerimpl.cpp \
    model/features/_impl/cellfunctionsensorimpl.cpp \
    model/features/_impl/cellfunctionweaponimpl.cpp \
    model/physics/codingphysicalquantities.cpp
HEADERS += \
    model/physics/physics.h \
    global/global.h \
    gui/monitoring/simulationmonitor.h \
    gui/microeditor/hexedit.h \
    gui/macroeditor/pixeluniverse.h \
    gui/macroeditor/shapeuniverse.h \
    gui/macroeditor/aliencellgraphicsitem.h \
    gui/macroeditor/aliencellconnectiongraphicsitem.h \
    gui/macroeditor/alienenergygraphicsitem.h \
    gui/microeditor/computercodeedit.h \
    gui/editorsettings.h \
    gui/macroeditor.h \
    gui/mainwindow.h \
    gui/microeditor.h \
    gui/microeditor/clusteredit.h \
    gui/microeditor/tokenedit.h \
    gui/macroeditor/markergraphicsitem.h \
    gui/microeditor/celledit.h \
    gui/microeditor/energyedit.h \
    gui/microeditor/tokentab.h \
    gui/microeditor/symboledit.h \
    gui/guisettings.h \
    gui/microeditor/metadatapropertiesedit.h \
    gui/microeditor/metadataedit.h \
    gui/dialogs/newsimulationdialog.h \
    gui/dialogs/simulationparametersdialog.h \
    gui/dialogs/addenergydialog.h \
    gui/dialogs/symboltabledialog.h \
    gui/dialogs/selectionmultiplyrandomdialog.h \
    gui/dialogs/selectionmultiplyarrangementdialog.h \
    gui/microeditor/cellcomputeredit.h \
    gui/dialogs/addrectstructuredialog.h \
    gui/dialogs/addhexagonstructuredialog.h \
    gui/assistance/tutorialwindow.h \
    model/simulationsettings.h \
    model/metadatamanager.h \
    gui/misc/startscreencontroller.h \
    global/servicelocator.h \
    model/features/constants.h \
    model/entities/entityfactory.h \
    model/entities/_impl/entityfactoryimpl.h \
    model/entities/cell.h \
    model/entities/cellto.h \
    model/entities/token.h \
    model/entities/grid.h \
    model/entities/energyparticle.h \
    model/entities/_impl/cellclusterimpl.h \
    model/entities/_impl/cellimpl.h \
    model/features/cellfeature.h \
    model/features/cellfeaturefactory.h \
    model/features/energyguidance.h \
    model/features/cellfunctioncomputer.h \
    model/features/cellfunction.h \
    model/features/_impl/energyguidanceimpl.h \
    model/features/_impl/cellfeaturefactoryimpl.h \
    model/simulationunit.h \
    model/simulationcontroller.h \
    model/entities/cellcluster.h \
    model/factoryfacade.h \
    model/_impl/factoryfacadeimpl.h \
    model/simulationcontext.h \
    model/definitions.h \
    model/_impl/simulationcontextimpl.h \
    model/topology.h \
    model/cellmap.h \
    model/energyparticlemap.h \
    model/serializationfacade.h \
    model/_impl/serializationfacadeimpl.h \
    model/features/cellfeatureconstants.h \
    model/features/_impl/cellfunctioncommunicatorimpl.h \
    model/features/_impl/cellfunctionconstructorimpl.h \
    model/features/_impl/cellfunctionpropulsionimpl.h \
    model/features/_impl/cellfunctionscannerimpl.h \
    model/features/_impl/cellfunctionsensorimpl.h \
    model/features/_impl/cellfunctionweaponimpl.h \
    model/physics/codingphysicalquantities.h \
    model/features/_impl/cellfunctioncomputerimpl.h
FORMS += gui/monitoring/simulationmonitor.ui \
    gui/macroeditor.ui \
    gui/mainwindow.ui \
    gui/microeditor/tokentab.ui \
    gui/microeditor/symboledit.ui \
    gui/microeditor/metadataedit.ui \
    gui/dialogs/newsimulationdialog.ui \
    gui/dialogs/simulationparametersdialog.ui \
    gui/dialogs/addenergydialog.ui \
    gui/dialogs/symboltabledialog.ui \
    gui/dialogs/selectionmultiplyrandomdialog.ui \
    gui/dialogs/selectionmultiplyarrangementdialog.ui \
    gui/microeditor/cellcomputeredit.ui \
    gui/dialogs/addrectstructuredialog.ui \
    gui/dialogs/addhexagonstructuredialog.ui \
    gui/assistance/tutorialwindow.ui

RESOURCES += \
    gui/resources/ressources.qrc

OTHER_FILES +=

test {
    message(Tests build)
    QT += testlib
    TARGET = Tests

    SOURCES -= main.cpp

    HEADERS += tests/predicates.h \
        tests/settings.h

    SOURCES += tests/testsuite.cpp \
        tests/predicates.cpp \
        tests/integrationtests/integrationtestreplicator.cpp \
        tests/integrationtests/integrationtestcomparison.cpp \
        tests/unittests/unittestphysics.spp \
        tests/unittests/unittestcellcluster.cpp \
        tests/unittests/unittestcellfunctioncommunicator.cpp

} else {
    message(Normal build)
}
