# Example: Demonstrating serializeOptionalNested() Backward Compatibility

This document demonstrates how `serializeOptionalNested()` maintains backward compatibility when adding new nested structures.

## Scenario

You have an existing structure:

```cpp
struct SensorGenomeDescription
{
    std::optional<int> autoTriggerInterval;
    int minRange;
    int maxRange;
    SensorModeGenomeDescription mode;  // variant with nested structures
};
```

Currently serialized as:

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
    
    ar(data._mode);  // Problem: No backward compatibility!
}
```

## Problem

If `SensorModeGenomeDescription` (the variant) adds a new alternative or its existing alternatives gain new fields, old save files won't be able to provide default values.

**Example issue:**
1. Old version has: `DetectEnergyGenomeDescription` with field `minDensity`
2. New version adds: `DetectEnergyGenomeDescription` with field `maxDensity`
3. Loading old save file: New field `maxDensity` is uninitialized (garbage value!)

## Solution with serializeOptionalNested()

### Step 1: Keep Existing Code (for now)

First deployment - no changes to existing structures to maintain compatibility:

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
    
    // Keep old serialization
    ar(data._mode);
}
```

### Step 2: For New Structures (or when updating)

When adding a new nested field or updating an existing structure:

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
    
    // NEW: Use optional nested serialization
    serializeOptionalNested(task, ar, data._mode, defaultObject._mode);
}
```

### What Happens

#### When Saving (new code)

Binary stream contains:
```
[LoadSave Map with 3 fields]
[true]  <- presence flag
[variant data...]
```

#### When Loading (new code, old file)

Old file contains:
```
[LoadSave Map with 3 fields]
[variant data without flag...]
```

**Result**: The `ar(hasData)` fails to find the boolean flag, so we need migration logic.

## Migration Strategy

To handle the transition from old format to new format:

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
        // Always save with new format
        serializeOptionalNested(task, ar, data._mode, defaultObject._mode);
    } else {
        // Try new format first
        try {
            serializeOptionalNested(task, ar, data._mode, defaultObject._mode);
        } catch (...) {
            // Fallback to old format for backward compatibility
            ar(data._mode);
        }
    }
}
```

## Better Approach: Version-Based Migration

Use the version information already in save files:

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
    
    // Check version (version is read at the start of deserialization)
    if (task == SerializationTask::Save || currentVersion >= VERSION_WITH_OPTIONAL_NESTED) {
        serializeOptionalNested(task, ar, data._mode, defaultObject._mode);
    } else {
        ar(data._mode);
    }
}
```

## Timeline Example

### Version 4.5.0 (Current)
```cpp
ar(data._mode);  // Direct serialization
```

**Save file format:**
```
Version: 4.5.0
[LoadSave Map]
[variant data]
```

### Version 4.6.0 (Transition)
```cpp
// Detect and handle both old and new formats
if (saving || version >= 4.6) {
    serializeOptionalNested(task, ar, data._mode, defaultObject._mode);
} else {
    ar(data._mode);
}
```

**New save files:**
```
Version: 4.6.0
[LoadSave Map]
[true][variant data]
```

**Can still load:** Version 4.5.0 files ✅

### Version 5.0.0 (Future)
```cpp
// Only use new format
serializeOptionalNested(task, ar, data._mode, defaultObject._mode);
```

**Save file format:**
```
Version: 5.0.0
[LoadSave Map]
[true][variant data]
```

**Can still load:** Version 4.6.0+ files ✅

## Real-World Example: Adding a New Field

### Scenario: Add `maxDensity` to `DetectEnergyGenomeDescription`

#### Before
```cpp
struct DetectEnergyGenomeDescription
{
    float minDensity = 1.0f;
};

// Serialized directly in SensorGenomeDescription
ar(data._mode);  // variant containing DetectEnergyGenomeDescription
```

#### After (with serializeOptionalNested)
```cpp
struct DetectEnergyGenomeDescription
{
    float minDensity = 1.0f;
    float maxDensity = 10.0f;  // NEW FIELD with default
};

// Now using optional nested serialization
serializeOptionalNested(task, ar, data._mode, defaultObject._mode);
```

#### Result
✅ **Old files load correctly**: 
- `minDensity` loads from file
- `maxDensity` uses default value (10.0f)

✅ **New files save correctly**:
- Both fields saved
- Presence flag ensures proper deserialization

## Key Benefits Demonstrated

1. **Additive Changes**: Can add new fields to nested structures safely
2. **Default Values**: Missing fields use sensible defaults from defaultObject
3. **Format Migration**: Clear path from old format to new format
4. **Backward Compatible**: Old files continue to work
5. **Forward Compatible**: New files work with new code

## Testing Strategy

### Test 1: Round-Trip with New Format
```cpp
TEST(SerializeOptionalNested, RoundTripNewFormat)
{
    SensorGenomeDescription original;
    original._mode = DetectEnergyGenomeDescription{._minDensity = 2.5f};
    
    // Save with new format
    std::stringstream stream;
    SerializationTask task = SerializationTask::Save;
    cereal::PortableBinaryOutputArchive ar(stream);
    // ... serialize ...
    
    // Load back
    SensorGenomeDescription loaded;
    // ... deserialize ...
    
    EXPECT_EQ(original._mode, loaded._mode);
}
```

### Test 2: Load Old Format
```cpp
TEST(SerializeOptionalNested, LoadOldFormat)
{
    // Simulate old save file without presence flag
    std::stringstream oldStream;
    // ... create old format data ...
    
    SensorGenomeDescription loaded;
    // ... deserialize with migration logic ...
    
    // Should load successfully with defaults for missing fields
    EXPECT_TRUE(loaded._mode != std::monostate{});
}
```

### Test 3: Default Values Applied
```cpp
TEST(SerializeOptionalNested, DefaultValuesApplied)
{
    std::stringstream stream;
    // Write only presence flag as false
    bool hasData = false;
    cereal::PortableBinaryOutputArchive(stream)(hasData);
    
    SensorGenomeDescription loaded;
    SensorGenomeDescription defaultObject;
    
    cereal::PortableBinaryInputArchive ar(stream);
    serializeOptionalNested(SerializationTask::Load, ar, loaded._mode, defaultObject._mode);
    
    // Should use default value
    EXPECT_EQ(loaded._mode, defaultObject._mode);
}
```

## Conclusion

The `serializeOptionalNested()` helper provides a practical solution for maintaining backward compatibility when adding fields to nested structures. By following the migration strategies outlined here, you can safely evolve your data structures while preserving compatibility with existing save files.

**Key Takeaway**: Start using `serializeOptionalNested()` for all new nested structures, and migrate existing ones when they need updates.
