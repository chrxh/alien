#pragma once

#include <QTextEdit>
#include <QTabWidget>

#include "Gui/Definitions.h"
#include "Gui/Settings.h"

class TabWidgetHelper
{
public:
	static void setupTextEdit(QTextEdit* tab)
	{
		tab->setFrameShape(QFrame::NoFrame);
		tab->setLineWidth(0);
		tab->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		tab->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		tab->setOverwriteMode(false);
		tab->setCursorWidth(6);
		tab->setPalette(GuiSettings::getPaletteForTab());
	}

	static void setupTabWidget(QTabWidget* tabWidget, QSize const& size)
	{
		tabWidget->setMinimumSize(size);
		tabWidget->setMaximumSize(size);
		tabWidget->setTabShape(QTabWidget::Triangular);
		tabWidget->setElideMode(Qt::ElideNone);
		tabWidget->setTabsClosable(false);
		tabWidget->setPalette(GuiSettings::getPaletteForTabWidget());
	}
};