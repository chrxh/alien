#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationAccess.h"

#include "Definitions.h"

class DataRepository : public QObject
{
    Q_OBJECT
public:
    DataRepository(QObject* parent = nullptr)
        : QObject(parent)
    {}
    virtual ~DataRepository() = default;

    virtual void
    init(Notifier* notifier, SimulationAccess* access, DescriptionHelper* connector, SimulationContext* context);

    virtual DataDescription& getDataRef();
    virtual CellDescription& getCellDescRef(uint64_t cellId);
    virtual ClusterDescription& getClusterDescRef(uint64_t cellId);
    virtual ClusterDescription const& getClusterDescRef(uint64_t cellId) const;
    virtual ParticleDescription& getParticleDescRef(uint64_t particleId);
    virtual ParticleDescription const& getParticleDescRef(uint64_t particleId) const;

    virtual void setSelectedTokenIndex(boost::optional<uint> const& value);
    virtual boost::optional<uint> getSelectedTokenIndex() const;

    virtual void addAndSelectCell(QVector2D const& posDelta);
    virtual void addAndSelectParticle(QVector2D const& posDelta);
    enum class Reconnect
    {
        No,
        Yes
    };
    virtual void addAndSelectData(DataDescription data, QVector2D const& posDelta, Reconnect reconnect = Reconnect::No);
    struct DataAndAngle
    {
        DataDescription data;
        boost::optional<double> angle;
    };
    virtual void addDataAtFixedPosition(vector<DataAndAngle> dataAndAngles);
    virtual void addRandomParticles(double totalEnergy, double maxEnergyPerParticle);
    virtual void deleteSelection();
    virtual void deleteExtendedSelection();
    virtual void addToken();
    virtual void addToken(TokenDescription const& token);
    virtual void deleteToken();

    virtual void setSelection(list<uint64_t> const& cellIds, list<uint64_t> const& particleIds);
    virtual void moveSelection(QVector2D const& delta);
    virtual void moveExtendedSelection(QVector2D const& delta);
    virtual void reconnectSelectedCells();
    virtual void rotateSelection(double angle);
    virtual void colorizeSelection(int colorCode);

    virtual void updateCluster(ClusterDescription const& cluster);
    virtual void updateParticle(ParticleDescription const& particle);

    virtual bool isInSelection(list<uint64_t> const& ids) const;
    virtual bool isInSelection(uint64_t id) const;  //id can mean cell or particle id
    virtual bool isInExtendedSelection(uint64_t id) const;
    virtual bool areEntitiesSelected() const;
    virtual unordered_set<uint64_t> getSelectedCellIds() const;
    virtual unordered_set<uint64_t> getSelectedParticleIds() const;
    virtual DataDescription getExtendedSelection() const;
    virtual bool isCellPresent(uint64_t cellId);

    virtual void requireDataUpdateFromSimulation(IntRect const& rect);
    virtual void requirePixelImageFromSimulation(IntRect const& rect, QImagePtr const& target);
    virtual void requireVectorImageFromSimulation(
        RealRect const& rect,
        double zoom,
        ImageResource const& image,
        IntVector2D const& imageSize);
    virtual std::mutex& getImageMutex();

    Q_SIGNAL void imageReady();


private:
    Q_SLOT void dataFromSimulationAvailable();
    Q_SLOT void sendDataChangesToSimulation(set<Receiver> const& targets);

    void updateAfterCellReconnections();
    void updateInternals(DataDescription const& data);
    bool isParticlePresent(uint64_t particleId);

    list<QMetaObject::Connection> _connections;

    Notifier* _notifier = nullptr;
    SimulationAccess* _access = nullptr;
    DescriptionHelper* _descHelper = nullptr;
    SimulationParameters _parameters;
    NumberGenerator* _numberGenerator = nullptr;
    DataDescription _data;
    DataDescription _unchangedData;

    boost::optional<uint> _selectedTokenIndex;
    unordered_set<uint64_t> _selectedCellIds;
    unordered_set<uint64_t> _selectedClusterIds;
    unordered_set<uint64_t> _selectedParticleIds;

    DescriptionNavigator _navi;
    RealRect _rect;
    IntVector2D _universeSize;
    std::mutex _mutex;
};
