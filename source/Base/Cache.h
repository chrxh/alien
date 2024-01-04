#pragma once

#include <unordered_map>
#include <optional>

template<typename Key, typename Value, int MaxEntries>
class Cache
{
public:
    void insert(Key const& key, Value const& value);

    std::optional<Value> find(Key const& key);

private:
    std::unordered_map<Key, Value> _cacheMap;
    std::list<Key> _usedKeys;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
template <typename Key, typename Value, int MaxEntries>
void Cache<Key, Value, MaxEntries>::insert(Key const& key, Value const& value)
{
    if (_cacheMap.size() >= MaxEntries) {
        _cacheMap.erase(_usedKeys.front());
        _usedKeys.pop_front();
    }
    try {
        auto keyInserted = _cacheMap.insert_or_assign(key, value).second;
        if (keyInserted) {
            _usedKeys.emplace_back(key);
        }
    } catch (...) {
    }
}

template <typename Key, typename Value, int MaxEntries>
std::optional<Value> Cache<Key, Value, MaxEntries>::find(Key const& key)
{
    auto findResult = _cacheMap.find(key);
    if (findResult != _cacheMap.end()) {
        return findResult->second;
    } else {
        return std::nullopt;
    }
}
