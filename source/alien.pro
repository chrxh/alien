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
    model/entities/impl/aliencellimpl.cpp \
    model/entities/aliencellcluster.cpp \
    model/physics/physics.cpp \
    global/global.cpp \
    model/entities/alientoken.cpp \
    model/entities/aliengrid.cpp \
    model/decorators/aliencellfunction.cpp \
    model/decorators/impl/aliencellfunctionscanner.cpp \
    model/entities/alienenergy.cpp \
    gui/monitoring/simulationmonitor.cpp \
    gui/microeditor/hexedit.cpp \
    model/alienthread.cpp \
    model/decorators/impl/aliencellfunctionconstructor.cpp \
    gui/macroeditor/pixeluniverse.cpp \
    gui/macroeditor/shapeuniverse.cpp \
    gui/macroeditor/aliencellgraphicsitem.cpp \
    gui/macroeditor/aliencellconnectiongraphicsitem.cpp \
    gui/macroeditor/alienenergygraphicsitem.cpp \
    gui/microeditor/computercodeedit.cpp \
    gui/macroeditor.cpp \
    gui/mainwindow.cpp \
    gui/microeditor.cpp \
    model/entities/aliencellto.cpp \
    model/aliensimulator.cpp \
    gui/microeditor/clusteredit.cpp \
    gui/microeditor/tokenedit.cpp \
    gui/macroeditor/markergraphicsitem.cpp \
    model/decorators/impl/aliencellfunctionweapon.cpp \
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
    model/decorators/impl/aliencellfunctionsensor.cpp \
    model/decorators/impl/aliencellfunctionpropulsion.cpp \
    gui/dialogs/addrectstructuredialog.cpp \
    gui/dialogs/addhexagonstructuredialog.cpp \
    gui/assistance/tutorialwindow.cpp \
    model/decorators/impl/aliencellfunctioncommunicator.cpp \
    model/simulationsettings.cpp \
    model/metadatamanager.cpp \
    gui/misc/startscreencontroller.cpp \
    gui/guisettings.cpp \
    model/entities/impl/aliencellfactoryimpl.cpp \
    model/decorators/impl/aliencelldecoratorfactoryimpl.cpp \
    global/servicelocator.cpp \
    model/decorators/impl/alienenergyguidancedecoratorimpl.cpp \
    model/decorators/impl/aliencellfunctioncomputerimpl.cpp \
    model/decorators/aliencellfunctioncomputer.cpp
HEADERS += \
    model/entities/impl/aliencellimpl.h \
    model/entities/aliencellcluster.h \
    model/physics/physics.h \
    global/global.h \
    model/entities/alientoken.h \
    model/entities/aliengrid.h \
    model/decorators/aliencelldecoratorfactory.h \
    model/decorators/aliencellfunctionscanner.h \
    model/entities/alienenergy.h \
    gui/monitoring/simulationmonitor.h \
    model/alienthread.h \
    model/decorators/impl/aliencellfunctionconstructor.h \
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
    model/entities/aliencellto.h \
    model/aliensimulator.h \
    gui/microeditor/clusteredit.h \
    gui/microeditor/tokenedit.h \
    gui/macroeditor/markergraphicsitem.h \
    model/decorators/impl/aliencellfunctionweapon.h \
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
    model/decorators/impl/aliencellfunctionsensor.h \
    model/decorators/impl/aliencellfunctionpropulsion.h \
    gui/dialogs/addrectstructuredialog.h \
    gui/dialogs/addhexagonstructuredialog.h \
    gui/assistance/tutorialwindow.h \
    model/decorators/impl/aliencellfunctioncommunicator.h \
    model/simulationsettings.h \
    model/metadatamanager.h \
    gui/misc/startscreencontroller.h \
    model/entities/aliencell.h \
    model/entities/aliencellfactory.h \
    model/entities/impl/aliencellfactoryimpl.h \
    model/decorators/impl/aliencelldecoratorfactoryimpl.h \
    global/servicelocator.h \
    model/decorators/aliencelldecorator.h \
    model/decorators/aliencellfunctioncomputer.h \
    model/decorators/aliencellfunction.h \
    model/decorators/alienenergyguidance.h \
    model/decorators/impl/alienenergyguidanceimpl.h \
    model/decorators/impl/aliencellfunctioncomputerimpl.h
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
