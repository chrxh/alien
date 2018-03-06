#include <QScrollBar>

#include "Gui/Settings.h"

#include "DataEditModel.h"
#include "DataEditController.h"
#include "MetadataEditWidget.h"
#include "MetadataEditTab.h"
#include "ui_MetadataEditTab.h"

MetadataEditTab::MetadataEditTab(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::MetadataEditTab)
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

void MetadataEditTab::init(DataEditModel * model, DataEditController * controller)
{
	_model = model;
	_controller = controller;
	ui->metadataEditWidget->init(model, controller);
}

void MetadataEditTab::updateDisplay ()
{
	auto const& metadata = *_model->getCellToEditRef().metadata;
	ui->metadataEditWidget->updateDisplay();
    ui->metadataDescriptionEdit->setText(metadata.description);
    if( ui->metadataDescriptionEdit->verticalScrollBar() )
        ui->metadataDescriptionEdit->verticalScrollBar()->setValue(0);
}

void MetadataEditTab::changesFromMetadataDescriptionEditor()
{
	auto& cell = *_model->getCellToEditRef().metadata;
	cell.description = ui->metadataDescriptionEdit->toPlainText();
	_controller->notificationFromMetadataTab();
}
