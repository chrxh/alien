#include "Gui/Settings.h"

#include "DataEditModel.h"
#include "SelectionEditTab.h"

SelectionEditTab::SelectionEditTab(QWidget * parent /*= nullptr*/) : QTextEdit(parent)
{
	
}

void SelectionEditTab::init(DataEditModel * model, DataEditController * controller)
{
	_model = model;
	_controller = controller;
}

void SelectionEditTab::updateDisplay()
{
	QString parStart = "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">";
	QString parEnd = "</p>";
	QString colorTextStart = "<span style=\"color:" + CELL_EDIT_TEXT_COLOR1.name() + "\">";
	QString colorDataStart = "<span style=\"color:" + CELL_EDIT_DATA_COLOR1.name() + "\">";
	QString colorEnd = "</span>";
	QString text;
	text = parStart + colorTextStart + "cells: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + colorEnd
		+ colorDataStart + QString("%1").arg(_model->getNumCells()) + colorEnd + parEnd;
	text += parStart + colorTextStart + "energy particles: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + colorEnd
		+ colorDataStart + QString("%1").arg(_model->getNumParticles()) + colorEnd + parEnd;
	setText(text);
}
