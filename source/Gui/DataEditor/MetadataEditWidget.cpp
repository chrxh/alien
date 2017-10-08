#include <QScrollBar>

#include "Gui/Settings.h"

#include "DataEditorModel.h"
#include "DataEditorController.h"
#include "MetadataPropertiesEditWidget.h"
#include "MetadataEditWidget.h"
#include "ui_MetadataEditWidget.h"

MetadataEditWidget::MetadataEditWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MetadataEditWidget)
{
    ui->setupUi(this);

    QPalette p = ui->cellDescriptionLabel->palette();
    p.setColor(QPalette::WindowText, CELL_EDIT_CAPTION_COLOR1);
    ui->cellDescriptionLabel->setPalette(p);

    p = ui->metadataDescriptionEdit->palette();
    p.setColor(QPalette::Text, CELL_EDIT_METADATA_CURSOR_COLOR);
    ui->metadataDescriptionEdit->setPalette(p);

    connect(ui->metadataDescriptionEdit, &QTextEdit::textChanged, this, &MetadataEditWidget::changesFromMetadataDescriptionEditor);
}

MetadataEditWidget::~MetadataEditWidget()
{
    delete ui;
}

void MetadataEditWidget::init(DataEditorModel * model, DataEditorController * controller)
{
	_model = model;
	_controller = controller;
	ui->metadataPropertiesEditWidget->init(model, controller);
}

void MetadataEditWidget::updateDisplay ()
{
	auto const& cell = *_model->getCellToEditRef().metadata;
	ui->metadataPropertiesEditWidget->updateDisplay();
    ui->metadataDescriptionEdit->setText(cell.description);
    if( ui->metadataDescriptionEdit->verticalScrollBar() )
        ui->metadataDescriptionEdit->verticalScrollBar()->setValue(0);
}

void MetadataEditWidget::changesFromMetadataDescriptionEditor()
{
	auto& cell = *_model->getCellToEditRef().metadata;
	cell.description = ui->metadataDescriptionEdit->toPlainText();
	_controller->notificationFromMetadataEditWidget();
}
