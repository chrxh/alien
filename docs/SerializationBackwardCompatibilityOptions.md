# Serialization Backward Compatibility Options for Nested Structures

## Problem Statement

The SerializerService uses a LoadSave map pattern (`std::unordered_map<int, std::variant>`) to maintain backward compatibility when deserializing data. When a key doesn't exist in the map, default values are used. This works well for top-level fields in structures.

However, nested structures that are serialized using cereal's `ar(...)` functions directly don't benefit from this backward compatibility mechanism. Examples include:
- `ar(data._mode)` - variant types containing nested genome descriptions
- `ar(data._nodes)` - vectors of complex structures
- `ar(data._connections, data._cellType, data._signal, ...)` - multiple nested objects

## Current Implementation Analysis

### LoadSave Map Pattern (Works Well)
```cpp
template <class Archive>
void loadSave(SerializationTask task, Archive& ar, SensorGenomeDescription& data)
{
    SensorGenomeDescription defaultObject;
    auto auxiliaries = getLoadSaveMap(task, ar);
    loadSave(task, auxiliaries, Id_SensorGenome_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
    loadSave(task, auxiliaries, Id_SensorGenome_MinRange, data._minRange, defaultObject._minRange);
    loadSave(task, auxiliaries, Id_SensorGenome_MaxRange, data._maxRange, defaultObject._maxRange);
    processLoadSaveMap(task, ar, auxiliaries);
    
    ar(data._mode);  // <-- Problem: No backward compatibility for nested variant!
}
```

### How LoadSave Map Works
1. **Save**: Collects all fields into a map with integer keys, then serializes the map
2. **Load**: Deserializes the map, then extracts values by key. Missing keys = use default values
3. **Benefit**: Adding new fields with defaults maintains backward compatibility with old files

### The Problem with ar() for Nested Structures
When `ar(data._mode)` is called:
- Cereal serializes the variant directly using its built-in serialization
- No map is created, no keys are used
- If the variant's alternative types gain new fields, old files can't provide defaults
- If new variant alternatives are added, deserialization may fail on old files

## Evaluated Options

### Option 1: Extend LoadSave Map to All Nested Structures

**Description**: Treat every nested complex structure (variants, vectors, nested objects) the same way as top-level fields - serialize them into the LoadSave map.

**Implementation**:
```cpp
// Add IDs for nested structures
auto constexpr Id_SensorGenome_Mode = 3;

template <class Archive>
void loadSave(SerializationTask task, Archive& ar, SensorGenomeDescription& data)
{
    SensorGenomeDescription defaultObject;
    auto auxiliaries = getLoadSaveMap(task, ar);
    loadSave(task, auxiliaries, Id_SensorGenome_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
    loadSave(task, auxiliaries, Id_SensorGenome_MinRange, data._minRange, defaultObject._minRange);
    loadSave(task, auxiliaries, Id_SensorGenome_MaxRange, data._maxRange, defaultObject._maxRange);
    loadSave(task, auxiliaries, Id_SensorGenome_Mode, data._mode, defaultObject._mode);  // <-- Changed!
    processLoadSaveMap(task, ar, auxiliaries);
}
```

**Pros**:
- Consistent approach across all serialization
- Maximum backward compatibility - all fields get default values if missing
- Easy to understand and maintain

**Cons**:
- VariantData type needs expansion to support more complex types (variants, vectors)
- Significant refactoring required across many structures
- May increase serialization overhead (storing more metadata)
- Complex types in variants may not fit well in VariantData std::variant

**Complexity**: High
**Backward Compatibility**: Excellent
**Recommended for**: Projects requiring maximum backward compatibility

---

### Option 2: Use Cereal's Optional Serialization with Versioning

**Description**: Leverage cereal's built-in features for optional fields and version-based serialization.

**Implementation**:
```cpp
template <class Archive>
void serialize(Archive& ar, SensorGenomeDescription& data, const std::uint32_t version)
{
    ar(data._autoTriggerInterval, data._minRange, data._maxRange);
    
    if (version >= 2) {
        ar(data._mode);
    } else {
        data._mode = SensorModeGenomeDescription();  // Default for old versions
    }
}
CEREAL_CLASS_VERSION(SensorGenomeDescription, 2);
```

**Pros**:
- Uses cereal's standard mechanisms
- Less custom code to maintain
- Version numbers are self-documenting
- Works well with cereal's ecosystem

**Cons**:
- Requires version tracking for each structure
- Less flexible than key-based approach
- Can't easily skip individual fields (version applies to whole structure)
- Adding a field in the middle requires careful version management

**Complexity**: Medium
**Backward Compatibility**: Good
**Recommended for**: Projects using cereal idiomatically

---

### Option 3: Hybrid Approach - LoadSave Map for Primitives, Optional Wrappers for Complex Types

**Description**: Keep the LoadSave map for primitive fields but wrap complex nested structures (variants, vectors) in optional wrappers that provide default values.

**Implementation**:
```cpp
template <class Archive>
void loadSave(SerializationTask task, Archive& ar, SensorGenomeDescription& data)
{
    SensorGenomeDescription defaultObject;
    auto auxiliaries = getLoadSaveMap(task, ar);
    loadSave(task, auxiliaries, Id_SensorGenome_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
    loadSave(task, auxiliaries, Id_SensorGenome_MinRange, data._minRange, defaultObject._minRange);
    loadSave(task, auxiliaries, Id_SensorGenome_MaxRange, data._maxRange, defaultObject._maxRange);
    processLoadSaveMap(task, ar, auxiliaries);
    
    // Wrap complex type in try-catch with default fallback
    try {
        ar(data._mode);
    } catch (...) {
        data._mode = defaultObject._mode;
    }
}
```

**Pros**:
- Minimal changes to existing code
- Keeps simple types using efficient LoadSave map
- Provides basic safety for complex types
- Easy to implement incrementally

**Cons**:
- Relies on exceptions for control flow (generally discouraged)
- Less precise than key-based approach
- Entire nested structure fails together (not field-by-field)
- Exception handling has performance overhead

**Complexity**: Low
**Backward Compatibility**: Fair
**Recommended for**: Quick wins with minimal refactoring

---

### Option 4: Optional Serialization with Has-Value Flags

**Description**: Serialize a boolean flag before each complex nested structure indicating whether it exists in the stream.

**Implementation**:
```cpp
template <class Archive>
void loadSave(SerializationTask task, Archive& ar, SensorGenomeDescription& data)
{
    SensorGenomeDescription defaultObject;
    auto auxiliaries = getLoadSaveMap(task, ar);
    loadSave(task, auxiliaries, Id_SensorGenome_AutoTriggerInterval, data._autoTriggerInterval, defaultObject._autoTriggerInterval);
    loadSave(task, auxiliaries, Id_SensorGenome_MinRange, data._minRange, defaultObject._minRange);
    loadSave(task, auxiliaries, Id_SensorGenome_MaxRange, data._maxRange, defaultObject._maxRange);
    processLoadSaveMap(task, ar, auxiliaries);
    
    if (task == SerializationTask::Save) {
        bool hasModeData = true;
        ar(hasModeData, data._mode);
    } else {
        bool hasModeData = false;
        ar(hasModeData);
        if (hasModeData) {
            ar(data._mode);
        } else {
            data._mode = defaultObject._mode;
        }
    }
}
```

**Pros**:
- Clean separation of presence vs. value
- No exception handling needed
- Works well for truly optional nested structures
- Explicit control over default values

**Cons**:
- Adds boolean overhead to serialization
- Boilerplate code for every nested structure
- Still doesn't handle individual fields within nested structures
- Changes serialization format (not backward compatible without migration)

**Complexity**: Medium
**Backward Compatibility**: Good (with migration)
**Recommended for**: New fields going forward

---

### Option 5: Recursive LoadSave Maps

**Description**: Apply the LoadSave map pattern recursively - nested structures get their own LoadSave maps.

**Implementation**:
```cpp
template <class Archive>
void loadSave(SerializationTask task, Archive& ar, SensorModeGenomeDescription& variant)
{
    // First serialize/deserialize the variant index
    std::size_t variantIndex = variant.index();
    ar(variantIndex);
    
    // Then apply LoadSave map to the active alternative
    std::visit([&](auto& activeAlternative) {
        using T = std::decay_t<decltype(activeAlternative)>;
        T defaultObject;
        auto auxiliaries = getLoadSaveMap(task, ar);
        // ... loadSave calls for fields of active alternative
        processLoadSaveMap(task, ar, auxiliaries);
    }, variant);
}
```

**Pros**:
- Extends backward compatibility to all levels
- Consistent approach throughout codebase
- Best long-term maintainability

**Cons**:
- Most complex to implement
- Requires custom serialization for variants and vectors
- Significant initial development effort
- May complicate debugging

**Complexity**: Very High
**Backward Compatibility**: Excellent
**Recommended for**: Critical systems requiring maximum flexibility

---

## Recommendation

### ✅ **IMPLEMENTED: Option 2 (Use Cereal's Optional Serialization with Versioning)**

**Rationale**:
1. **Uses cereal idioms**: Leverages cereal's built-in versioning system
2. **Less custom code**: Works with cereal's standard mechanisms
3. **Self-documenting**: Version numbers make it clear when fields were added
4. **Integrates with LoadSave**: Combined with existing LoadSave map pattern for primitives

**Implementation**:
The implementation uses cereal's versioned `serialize` function for structures with nested data, while keeping the LoadSave map pattern for primitive fields. This provides backward compatibility for both primitive and nested fields.

```cpp
template <class Archive>
void serialize(Archive& ar, SensorGenomeDescription& data, std::uint32_t const version)
{
    SensorGenomeDescription defaultObject;
    SerializationTask task = Archive::is_loading::value ? SerializationTask::Load : SerializationTask::Save;
    auto auxiliaries = getLoadSaveMap(task, ar);
    // ... LoadSave map for primitive fields
    processLoadSaveMap(task, ar, auxiliaries);
    
    // Version 1: Added _mode field
    if (version >= 1) {
        ar(data._mode);
    } else if (task == SerializationTask::Load) {
        data._mode = defaultObject._mode;
    }
}
CEREAL_CLASS_VERSION(SensorGenomeDescription, 1);
```

**Structures Converted**:
- SensorGenomeDescription
- MuscleGenomeDescription  
- NodeDescription
- GeneDescription
- GenomeDescription
- SensorDescription
- MuscleDescription
- CellDescription
- CreatureDescription

All structures with nested data now use versioned serialization with `CEREAL_CLASS_VERSION` macros.

---

## Implementation Plan

### Phase 1: Helper Functions (Immediate)
- Create `serializeOptional` helper template
- Add to SerializerService.cpp
- Document usage pattern

### Phase 2: Apply to New Code (Ongoing)
- Use helper for any new nested structures
- Document in coding guidelines

### Phase 3: Gradual Migration (As Needed)
- Convert existing nested structures when they need updates
- Maintain backward compatibility with migration code

### Phase 4: Future Enhancement (Next Major Version)
- Consider full recursive LoadSave map implementation
- Plan migration strategy for old save files

---

## Testing Strategy

For any chosen option, comprehensive testing is required:

1. **Backward Compatibility Tests**:
   - Save with old version, load with new version
   - Verify default values applied correctly

2. **Forward Compatibility Tests**:
   - Save with new version (with new fields), load with old version
   - Verify graceful handling

3. **Round-Trip Tests**:
   - Save → Load → Save → Load
   - Verify data integrity maintained

4. **Edge Cases**:
   - Empty collections
   - Variant with all alternatives
   - Deeply nested structures

---

## Conclusion

The current LoadSave map pattern provides excellent backward compatibility for primitive fields. Extending this to nested structures requires careful consideration of complexity vs. benefit trade-offs.

**Recommended immediate action**: Implement Option 4 (Optional Serialization with Has-Value Flags) for new nested structures, with a long-term plan to migrate to Option 5 (Recursive LoadSave Maps) in a future major version.

This provides a practical balance between immediate improvements and long-term maintainability.
