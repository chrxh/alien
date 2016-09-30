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
    globaldata/globalfunctions.cpp \
    simulation/entities/alientoken.cpp \
    simulation/entities/aliengrid.cpp \
    simulation/processing/aliencellfunction.cpp \
    simulation/processing/aliencellfunctionfactory.cpp \
    simulation/processing/aliencellfunctionscanner.cpp \
    simulation/entities/alienenergy.cpp \
    graphics/monitoring/simulationmonitor.cpp \
    graphics/microeditor/hexedit.cpp \
    simulation/processing/alienthread.cpp \
    simulation/processing/aliencellfunctionconstructor.cpp \
    graphics/macroeditor/pixeluniverse.cpp \
    graphics/macroeditor/shapeuniverse.cpp \
    graphics/macroeditor/aliencellgraphicsitem.cpp \
    graphics/macroeditor/aliencellconnectiongraphicsitem.cpp \
    graphics/macroeditor/alienenergygraphicsitem.cpp \
    graphics/microeditor/computercodeedit.cpp \
    graphics/macroeditor.cpp \
    graphics/mainwindow.cpp \
    graphics/microeditor.cpp \
    simulation/aliencellreduced.cpp \
    simulation/aliensimulator.cpp \
    graphics/microeditor/clusteredit.cpp \
    graphics/microeditor/tokenedit.cpp \
    graphics/macroeditor/markergraphicsitem.cpp \
    simulation/processing/aliencellfunctioncomputer.cpp \
    simulation/processing/aliencellfunctionweapon.cpp \
    graphics/microeditor/celledit.cpp \
    graphics/microeditor/energyedit.cpp \
    graphics/microeditor/tokentab.cpp \
    graphics/microeditor/symboledit.cpp \
    graphics/microeditor/metadatapropertiesedit.cpp \
    graphics/microeditor/metadataedit.cpp \
    graphics/dialogs/newsimulationdialog.cpp \
    graphics/dialogs/simulationparametersdialog.cpp \
    graphics/dialogs/addenergydialog.cpp \
    graphics/dialogs/symboltabledialog.cpp \
    graphics/dialogs/selectionmultiplyrandomdialog.cpp \
    graphics/dialogs/selectionmultiplyarrangementdialog.cpp \
    graphics/microeditor/cellcomputeredit.cpp \
    simulation/processing/aliencellfunctionsensor.cpp \
    simulation/processing/aliencellfunctionpropulsion.cpp \
    graphics/dialogs/addrectstructuredialog.cpp \
    graphics/dialogs/addhexagonstructuredialog.cpp \
    graphics/assistance/tutorialwindow.cpp \
    simulation/entities/testaliencellcluster.cpp \
    simulation/entities/testalientoken.cpp \
    simulation/processing/aliencellfunctioncommunicator.cpp \
    globaldata/simulationsettings.cpp \
    simulation/metadatamanager.cpp
HEADERS += \
    simulation/entities/aliencell.h \
    simulation/entities/aliencellcluster.h \
    simulation/physics/physics.h \
    globaldata/globalfunctions.h \
    simulation/entities/alientoken.h \
    simulation/entities/aliengrid.h \
    simulation/processing/aliencellfunction.h \
    simulation/processing/aliencellfunctionfactory.h \
    simulation/processing/aliencellfunctionscanner.h \
    simulation/entities/alienenergy.h \
    graphics/monitoring/simulationmonitor.h \
    simulation/processing/alienthread.h \
    simulation/processing/aliencellfunctionconstructor.h \
    graphics/microeditor/hexedit.h \
    graphics/macroeditor/pixeluniverse.h \
    graphics/macroeditor/shapeuniverse.h \
    graphics/macroeditor/aliencellgraphicsitem.h \
    graphics/macroeditor/aliencellconnectiongraphicsitem.h \
    graphics/macroeditor/alienenergygraphicsitem.h \
    graphics/microeditor/computercodeedit.h \
    globaldata/editorsettings.h \
    graphics/macroeditor.h \
    graphics/mainwindow.h \
    graphics/microeditor.h \
    simulation/aliencellreduced.h \
    simulation/aliensimulator.h \
    graphics/microeditor/clusteredit.h \
    graphics/microeditor/tokenedit.h \
    graphics/macroeditor/markergraphicsitem.h \
    simulation/processing/aliencellfunctioncomputer.h \
    simulation/processing/aliencellfunctionweapon.h \
    graphics/microeditor/celledit.h \
    graphics/microeditor/energyedit.h \
    graphics/microeditor/tokentab.h \
    graphics/microeditor/symboledit.h \
    globaldata/guisettings.h \
    graphics/microeditor/metadatapropertiesedit.h \
    graphics/microeditor/metadataedit.h \
    graphics/dialogs/newsimulationdialog.h \
    graphics/dialogs/simulationparametersdialog.h \
    graphics/dialogs/addenergydialog.h \
    graphics/dialogs/symboltabledialog.h \
    graphics/dialogs/selectionmultiplyrandomdialog.h \
    graphics/dialogs/selectionmultiplyarrangementdialog.h \
    graphics/microeditor/cellcomputeredit.h \
    simulation/processing/aliencellfunctionsensor.h \
    simulation/processing/aliencellfunctionpropulsion.h \
    graphics/dialogs/addrectstructuredialog.h \
    graphics/dialogs/addhexagonstructuredialog.h \
    graphics/assistance/tutorialwindow.h \
    simulation/processing/aliencellfunctioncommunicator.h \
    globaldata/simulationsettings.h \
    simulation/metadatamanager.h
FORMS += graphics/monitoring/simulationmonitor.ui \
    graphics/macroeditor.ui \
    graphics/mainwindow.ui \
    graphics/microeditor/tokentab.ui \
    graphics/microeditor/symboledit.ui \
    graphics/microeditor/metadataedit.ui \
    graphics/dialogs/newsimulationdialog.ui \
    graphics/dialogs/simulationparametersdialog.ui \
    graphics/dialogs/addenergydialog.ui \
    graphics/dialogs/symboltabledialog.ui \
    graphics/dialogs/selectionmultiplyrandomdialog.ui \
    graphics/dialogs/selectionmultiplyarrangementdialog.ui \
    graphics/microeditor/cellcomputeredit.ui \
    graphics/dialogs/addrectstructuredialog.ui \
    graphics/dialogs/addhexagonstructuredialog.ui \
    graphics/assistance/tutorialwindow.ui

RESOURCES += \
    graphics/resources/ressources.qrc

OTHER_FILES +=
