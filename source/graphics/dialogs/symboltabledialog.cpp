#include "symboltabledialog.h"
#include "ui_symboltabledialog.h"

#include "../../globaldata/globalfunctions.h"
#include "../../globaldata/metadatamanager.h"

#include <QFileDialog>
#include <QMessageBox>

SymbolTableDialog::SymbolTableDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SymbolTableDialog)
{
    ui->setupUi(this);
    setFont(GlobalFunctions::getGlobalFont());

//    _localMeta.setSymbolTable(meta->getSymbolTable());

    //create headers
    ui->tableWidget->horizontalHeader()->resizeSection(0, 400);
    ui->tableWidget->horizontalHeaderItem(0)->setFont(GlobalFunctions::getGlobalFont());
    ui->tableWidget->horizontalHeaderItem(0)->setTextAlignment(Qt::AlignLeft);
    ui->tableWidget->horizontalHeader()->resizeSection(1, 100);
    ui->tableWidget->horizontalHeaderItem(1)->setFont(GlobalFunctions::getGlobalFont());
    ui->tableWidget->horizontalHeaderItem(1)->setTextAlignment(Qt::AlignLeft);

    setSymbolTableToWidget(&MetadataManager::getGlobalInstance());

    //connections
    connect(ui->tableWidget, SIGNAL(itemSelectionChanged()), this, SLOT(itemSelectionChanged()));
//    connect(ui->buttonBox, SIGNAL(accepted()), this, SLOT(getSymbolTableFromWidgets()));
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

void SymbolTableDialog::updateSymbolTable (MetadataManager* meta)
{
    meta->clearSymbolTable();
    for(int i = 0; i < ui->tableWidget->rowCount(); ++i) {
        QTableWidgetItem* item1 = ui->tableWidget->item(i, 0);
        QTableWidgetItem* item2 = ui->tableWidget->item(i, 1);
        meta->addSymbolEntry(item1->text(), item2->text());
    }
}

void SymbolTableDialog::setSymbolTableToWidget (MetadataManager* meta)
{
    int row = ui->tableWidget->rowCount();
    for(int i = 0; i < row; ++i)
        ui->tableWidget->removeRow(0);

    //create entries in the table
    QMapIterator< QString, QString > it = meta->getSymbolTable();
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
//        QString key = ui->tableWidget->item(row, 0)->text();
//        _meta->delSymbolEntry(key);
        ui->tableWidget->removeRow(row);
    }
    if( ui->tableWidget->rowCount() == 0 )
        ui->delButton->setEnabled(false);
}

void SymbolTableDialog::defaultButtonClicked ()
{
    MetadataManager* localMeta = new MetadataManager();
    setSymbolTableToWidget(localMeta);
    delete localMeta;
}

void SymbolTableDialog::loadButtonClicked ()
{
    MetadataManager* localMeta = new MetadataManager();
    QString fileName = QFileDialog::getOpenFileName(this, "Load Symbol Table", "", "Alien Symbol Table(*.sym)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {

            //read simulation data
            QDataStream in(&file);
            localMeta->readSymbolTable(in, false);
            setSymbolTableToWidget(localMeta);
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The specified symbol table could not loaded.");
            msgBox.exec();
        }
    }
    delete localMeta;
}

void SymbolTableDialog::saveButtonClicked ()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save Symbol Table", "", "Alien Symbol Table (*.sym)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::WriteOnly) ) {

            MetadataManager* localMeta = new MetadataManager();
            updateSymbolTable(localMeta);

            //serialize symbol table
            QDataStream out(&file);
            localMeta->serializeSymbolTable(out);
            file.close();
            delete localMeta;
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The symbol table could not saved.");
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

            //read simulation data
            MetadataManager* localMeta = new MetadataManager();
            updateSymbolTable(localMeta);
            QDataStream in(&file);
            localMeta->readSymbolTable(in, true);
            setSymbolTableToWidget(localMeta);
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The specified symbol table could not loaded.");
            msgBox.exec();
        }
    }
}
