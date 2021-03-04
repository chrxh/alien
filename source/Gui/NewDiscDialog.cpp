#include "NewDiscDialog.h"

#include <QMessageBox>

#include "Settings.h"
#include "StringHelper.h"

NewDiscDialog::NewDiscDialog(QWidget *parent)
    : QDialog(parent)
{
    ui.setupUi(this);

    ui.outerRadiusEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::NewCircleOuterRadiusKey, Const::NewCircleOuterRadiusDefault)));
    ui.innerRadiusEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::NewCircleInnerRadiusKey, Const::NewCircleInnerRadiusDefault)));
    ui.distEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::NewCircleDistanceKey, Const::NewCircleDistanceDefault)));
    ui.energyEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::NewCircleCellEnergyKey, Const::NewCircleCellEnergyDefault)));
    ui.colorCodeEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::NewCircleColorCodeKey, Const::NewCircleColorCodeDefault)));

	connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &NewDiscDialog::okClicked);
}

NewDiscDialog::~NewDiscDialog() {}

int NewDiscDialog::getOuterRadius() const
{
    bool ok(true);
    return ui.outerRadiusEdit->text().toInt(&ok);
}

int NewDiscDialog::getInnerRadius() const
{
    bool ok(true);
    return ui.innerRadiusEdit->text().toInt(&ok);
}

double NewDiscDialog::getDistance() const
{
    bool ok(true);
    return ui.distEdit->text().toDouble(&ok);
}

double NewDiscDialog::getCellEnergy() const
{
    bool ok(true);
    return ui.energyEdit->text().toDouble(&ok);
}

int NewDiscDialog::getColorCode() const
{
    bool ok(true);
    return ui.colorCodeEdit->text().toInt(&ok);
}

bool NewDiscDialog::validate() const
{
    auto result = true;
    auto ok = true;
    auto outerRadius = ui.outerRadiusEdit->text().toInt(&ok);
    result &= ok;
    auto innerRadius = ui.innerRadiusEdit->text().toInt(&ok);
    result &= ok;
    auto distance = ui.distEdit->text().toDouble(&ok);
    result &= ok;
    auto energy = ui.energyEdit->text().toDouble(&ok);
    result &= ok;
    auto colorCode = ui.colorCodeEdit->text().toInt(&ok);
    result &= ok;

    if (outerRadius < 0 || innerRadius < 0 || outerRadius < innerRadius || energy <= 0.0 || colorCode < 0
        || distance < 0.0) {
        return false;
    }

    return result;
}

void NewDiscDialog::okClicked()
{
    if (!validate()) {
        QMessageBox msgBox(QMessageBox::Critical, "Invalid values", Const::ErrorInvalidValues);
        msgBox.exec();
        return;
    }
    GuiSettings::setSettingsValue(Const::NewCircleOuterRadiusKey, getOuterRadius());
    GuiSettings::setSettingsValue(Const::NewCircleInnerRadiusKey, getInnerRadius());
    GuiSettings::setSettingsValue(Const::NewCircleDistanceKey, getDistance());
    GuiSettings::setSettingsValue(Const::NewCircleCellEnergyKey, getCellEnergy());
    GuiSettings::setSettingsValue(Const::NewCircleColorCodeKey, getColorCode());
    accept();
}
