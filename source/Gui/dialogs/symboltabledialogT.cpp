#include <QFileDialog>
#include <QMessageBox>

#include "Base/ServiceLocator.h"
#include "Model/Settings.h"
#include "Model/SerializationFacade.h"
#include "Model/Metadata/SymbolTable.h"
#include "Model/ModelBuilderFacade.h"
#include "gui/SettingsT.h"

#include "symboltabledialogT.h"
#include "ui_symboltabledialog.h"

SymbolTableDialog::SymbolTableDialog(SymbolTable* symbolTable, QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::SymbolTableDialog)
	, _symbolTable(symbolTable->clone())
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());

    //create headers
    ui->tableWidget->horizontalHeader()->resizeSection(0, 400);
    ui->tableWidget->horizontalHeaderItem(0)->setFont(GuiSettings::getGlobalFont());
    ui->tableWidget->horizontalHeaderItem(0)->setTextAlignment(Qt::AlignLeft);
    ui->tableWidget->horizontalHeader()->resizeSection(1, 100);
    ui->tableWidget->horizontalHeaderItem(1)->setFont(GuiSettings::getGlobalFont());
    ui->tableWidget->horizontalHeaderItem(1)->setTextAlignment(Qt::AlignLeft);

    symbolTableToWidgets();

    //connections
    connect(ui->tableWidget, SIGNAL(itemSelectionChanged()), this, SLOT(itemSelectionChanged()));
    connect(ui->addButton, SIGNAL(clicked()), this, SLOT(addButtonClicked()));
    connect(ui->delButton, SIGNAL(clicked()), this, SLOT(delButtonClicked()));
    connect(ui->defaultButton, SIGNAL(clicked()), this, SLOT(defaultButtonClicked()));
    connect(ui->loadButton, SIGNAL(clicked()), this, SLOT(loadButtonClicked()));
    connect(ui->saveButton, SIGNAL(clicked()), this, SLOT(saveButtonClicked()));
    connect(ui->mergeWithButton, SIGNAL(clicked()), this, SLOT(mergeWithButtonClicked()));
}

SymbolTableDialog::~SymbolTableDialog()
{
    delete ui;
}

SymbolTable* SymbolTableDialog::getNewSymbolTable()
{
	widgetsToSymbolTable();
	return _symbolTable;
}

void SymbolTableDialog::widgetsToSymbolTable ()
{
    _symbolTable->clearTable();
    for(int i = 0; i < ui->tableWidget->rowCount(); ++i) {
        QTableWidgetItem* item1 = ui->tableWidget->item(i, 0);
        QTableWidgetItem* item2 = ui->tableWidget->item(i, 1);
        _symbolTable->addEntry(item1->text(), item2->text());
    }
}

void SymbolTableDialog::symbolTableToWidgets()
{
    int row = ui->tableWidget->rowCount();
    for(int i = 0; i < row; ++i)
        ui->tableWidget->removeRow(0);

    //create entries in the table
    QMapIterator< QString, QString > it = _symbolTable->getTableConstRef();
    row = 0;
    while( it.hasNext() ) {
        it.next();

        //create new row in table
        ui->tableWidget->insertRow(row);
        ui->tableWidget->setVerticalHeaderItem(row, new QTableWidgetItem(""));
        QTableWidgetItem* item = new QTableWidgetItem(it.key());
        ui->tableWidget->setItem(row, 0, item);
        item = new QTableWidgetItem(it.value());
        ui->tableWidget->setItem(row, 1, item);
        row++;
    }
}

void SymbolTableDialog::itemSelectionChanged ()
{
    //items selected?
    if( ui->tableWidget->selectedItems().empty() ) {
        ui->delButton->setEnabled(false);
    }
    else {
        ui->delButton->setEnabled(true);
    }
}

void SymbolTableDialog::addButtonClicked ()
{
    //create new row in table
    int row = ui->tableWidget->rowCount();
    ui->tableWidget->insertRow(row);
    ui->tableWidget->setVerticalHeaderItem(row, new QTableWidgetItem(""));
    QTableWidgetItem* item = new QTableWidgetItem("");
    ui->tableWidget->setItem(row, 0, item);
    ui->tableWidget->setItem(row, 1, new QTableWidgetItem(""));
    ui->tableWidget->editItem(item);
    ui->tableWidget->setCurrentCell(row,0);

    //activate del button
    ui->delButton->setEnabled(true);
}

void SymbolTableDialog::delButtonClicked ()
{
    while( !ui->tableWidget->selectedItems().isEmpty() ) {
        QList< QTableWidgetItem* > items = ui->tableWidget->selectedItems();
        int row = items.at(0)->row();
        ui->tableWidget->removeRow(row);
    }
    if( ui->tableWidget->rowCount() == 0 )
        ui->delButton->setEnabled(false);
}

void SymbolTableDialog::defaultButtonClicked ()
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();

	_symbolTable = facade->buildDefaultSymbolTable();
    symbolTableToWidgets();
}

void SymbolTableDialog::loadButtonClicked ()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Load Symbol Table", "", "Alien Symbol Table(*.sym)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {
			QDataStream in(&file);

			SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();
			SymbolTable* symbolTable = facade->deserializeSymbolTable(in);
			_symbolTable = symbolTable;
			delete symbolTable;
            symbolTableToWidgets();
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The specified symbol table could not loaded.");
            msgBox.exec();
        }
    }
}

void SymbolTableDialog::saveButtonClicked ()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save Symbol Table", "", "Alien Symbol Table (*.sym)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::WriteOnly) ) {
			widgetsToSymbolTable();

			SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();
			QDataStream out(&file);
			facade->serializeSymbolTable(_symbolTable, out);
			file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The symbol table could not saved.");
            msgBox.exec();
        }
    }
}

void SymbolTableDialog::mergeWithButtonClicked ()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Load Symbol Table", "", "Alien Symbol Table(*.sym)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {
			widgetsToSymbolTable();

			QDataStream in(&file);
			SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();
			SymbolTable* symbolTable = facade->deserializeSymbolTable(in);
			_symbolTable->mergeTable(*symbolTable);
			delete symbolTable;
			file.close();

			symbolTableToWidgets();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The specified symbol table could not loaded.");
            msgBox.exec();
        }
    }
}
