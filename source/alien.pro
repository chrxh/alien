# -------------------------------------------------
# Project created by QtCreator 2012-04-08T22:05:32
# -------------------------------------------------
QT += opengl
QT += testlib
QMAKE_CXXFLAGS_RELEASE    = -O4 -march=native -ffast-math -funroll-loops -std=c++11
QMAKE_CXXFLAGS_DEBUG    = -std=c++11
TARGET = alien
TEMPLATE = app
SOURCES += main.cpp \
    simulation/entities/aliencell.cpp \
    simulation/entities/aliencellcluster.cpp \
    simulation/physics/physics.cpp \
    global/globalfunctions.cpp \
    simulation/entities/alientoken.cpp \
    simulation/entities/aliengrid.cpp \
    simulation/processing/aliencellfunction.cpp \
    simulation/processing/aliencellfunctionfactory.cpp \
    simulation/processing/aliencellfunctionscanner.cpp \
    simulation/entities/alienenergy.cpp \
    gui/monitoring/simulationmonitor.cpp \
    gui/microeditor/hexedit.cpp \
    simulation/processing/alienthread.cpp \
    simulation/processing/aliencellfunctionconstructor.cpp \
    gui/macroeditor/pixeluniverse.cpp \
    gui/macroeditor/shapeuniverse.cpp \
    gui/macroeditor/aliencellgraphicsitem.cpp \
    gui/macroeditor/aliencellconnectiongraphicsitem.cpp \
    gui/macroeditor/alienenergygraphicsitem.cpp \
    gui/microeditor/computercodeedit.cpp \
    gui/macroeditor.cpp \
    gui/mainwindow.cpp \
    gui/microeditor.cpp \
    simulation/aliencellreduced.cpp \
    simulation/aliensimulator.cpp \
    gui/microeditor/clusteredit.cpp \
    gui/microeditor/tokenedit.cpp \
    gui/macroeditor/markergraphicsitem.cpp \
    simulation/processing/aliencellfunctioncomputer.cpp \
    simulation/processing/aliencellfunctionweapon.cpp \
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
    simulation/processing/aliencellfunctionsensor.cpp \
    simulation/processing/aliencellfunctionpropulsion.cpp \
    gui/dialogs/addrectstructuredialog.cpp \
    gui/dialogs/addhexagonstructuredialog.cpp \
    gui/assistance/tutorialwindow.cpp \
    simulation/processing/aliencellfunctioncommunicator.cpp \
    global/simulationsettings.cpp \
    simulation/metadatamanager.cpp \
    gui/misc/startscreencontroller.cpp
HEADERS += \
    simulation/entities/aliencell.h \
    simulation/entities/aliencellcluster.h \
    simulation/physics/physics.h \
    global/globalfunctions.h \
    simulation/entities/alientoken.h \
    simulation/entities/aliengrid.h \
    simulation/processing/aliencellfunction.h \
    simulation/processing/aliencellfunctionfactory.h \
    simulation/processing/aliencellfunctionscanner.h \
    simulation/entities/alienenergy.h \
    gui/monitoring/simulationmonitor.h \
    simulation/processing/alienthread.h \
    simulation/processing/aliencellfunctionconstructor.h \
    gui/microeditor/hexedit.h \
    gui/macroeditor/pixeluniverse.h \
    gui/macroeditor/shapeuniverse.h \
    gui/macroeditor/aliencellgraphicsitem.h \
    gui/macroeditor/aliencellconnectiongraphicsitem.h \
    gui/macroeditor/alienenergygraphicsitem.h \
    gui/microeditor/computercodeedit.h \
    global/editorsettings.h \
    gui/macroeditor.h \
    gui/mainwindow.h \
    gui/microeditor.h \
    simulation/aliencellreduced.h \
    simulation/aliensimulator.h \
    gui/microeditor/clusteredit.h \
    gui/microeditor/tokenedit.h \
    gui/macroeditor/markergraphicsitem.h \
    simulation/processing/aliencellfunctioncomputer.h \
    simulation/processing/aliencellfunctionweapon.h \
    gui/microeditor/celledit.h \
    gui/microeditor/energyedit.h \
    gui/microeditor/tokentab.h \
    gui/microeditor/symboledit.h \
    global/guisettings.h \
    gui/microeditor/metadatapropertiesedit.h \
    gui/microeditor/metadataedit.h \
    gui/dialogs/newsimulationdialog.h \
    gui/dialogs/simulationparametersdialog.h \
    gui/dialogs/addenergydialog.h \
    gui/dialogs/symboltabledialog.h \
    gui/dialogs/selectionmultiplyrandomdialog.h \
    gui/dialogs/selectionmultiplyarrangementdialog.h \
    gui/microeditor/cellcomputeredit.h \
    simulation/processing/aliencellfunctionsensor.h \
    simulation/processing/aliencellfunctionpropulsion.h \
    gui/dialogs/addrectstructuredialog.h \
    gui/dialogs/addhexagonstructuredialog.h \
    gui/assistance/tutorialwindow.h \
    simulation/processing/aliencellfunctioncommunicator.h \
    global/simulationsettings.h \
    simulation/metadatamanager.h \
    gui/misc/startscreencontroller.h
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
    TARGET = UnitTests

    SOURCES -= main.cpp

    HEADERS += tests/testphysics.h \
        tests/testaliencellcluster.h \
        tests/testalientoken.h \
        tests/testsettings.h \
        tests/testaliencellfunctioncommunicator.h


    SOURCES += tests/testsuite.cpp

} else {
    message(Normal build)
}
