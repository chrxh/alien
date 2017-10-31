#include <QScrollBar>

#include "Gui/Settings.h"

#include "DataEditorModel.h"
#include "DataEditorController.h"
#include "MetadataEditWidget.h"
#include "MetadataEditTab.h"
#include "ui_MetadataEditTab.h"

MetadataEditTab::MetadataEditTab(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MetadataEditTab)
{
    ui->setupUi(this);

    QPalette p = ui->cellDescriptionLabel->palette();
    p.setColor(QPalette::WindowText, CELL_EDIT_CAPTION_COLOR1);
    ui->cellDescriptionLabel->setPalette(p);

    p = ui->metadataDescriptionEdit->palette();
    p.setColor(QPalette::Text, CELL_EDIT_METADATA_CURSOR_COLOR);
    ui->metadataDescriptionEdit->setPalette(p);

    connect(ui->metadataDescriptionEdit, &QTextEdit::textChanged, this, &MetadataEditTab::changesFromMetadataDescriptionEditor);
}

MetadataEditTab::~MetadataEditTab()
{
    delete ui;
}

void MetadataEditTab::init(DataEditorModel * model, DataEditorController * controller)
{
	_model = model;
	_controller = controller;
	ui->metadataPropertiesEditWidget->init(model, controller);
}

void MetadataEditTab::updateDisplay ()
{
	auto const& cell = *_model->getCellToEditRef().metadata;
	ui->metadataPropertiesEditWidget->updateDisplay();
    ui->metadataDescriptionEdit->setText(cell.description);
    if( ui->metadataDescriptionEdit->verticalScrollBar() )
        ui->metadataDescriptionEdit->verticalScrollBar()->setValue(0);
}

void MetadataEditTab::changesFromMetadataDescriptionEditor()
{
	auto& cell = *_model->getCellToEditRef().metadata;
	cell.description = ui->metadataDescriptionEdit->toPlainText();
	_controller->notificationFromMetadataEditWidget();
}
