
DATASET_DIR="datasets"
TOOLS_DIR="tools"
EXTRACT_SCRIPT="extract_frames_uavid.py"
ORGANIZE_SCRIPT="organize_uavid.py"

UAVID_ZIP="$DATASET_DIR/uavid_v1.5_official_release.zip"
VDD_ZIP="$DATASET_DIR/VDD.zip"

if [ -f "$UAVID_ZIP" ]; then
    echo "UAVid ZIP file found. Extracting..."
    mkdir -p "$DATASET_DIR/uavid_extracted"
    unzip -o "$UAVID_ZIP" -d "$DATASET_DIR/uavid_extracted"
    
    if [ -d "$DATASET_DIR/uavid_extracted/uavid_v1.5_official_release/uavid_v1.5_official_release" ]; then
        echo "Fixing nested directory structure..."
        
        mv "$DATASET_DIR/uavid_extracted/uavid_v1.5_official_release/uavid_v1.5_official_release"/* "$DATASET_DIR/uavid_extracted/uavid_v1.5_official_release/"
        
        rmdir "$DATASET_DIR/uavid_extracted/uavid_v1.5_official_release/uavid_v1.5_official_release"
    fi
fi

if [ -f "$VDD_ZIP" ]; then
    echo "VDD ZIP file found. Extracting..."
    unzip -o "$VDD_ZIP" -d "$DATASET_DIR/VDD"
    
        if [ -d "$DATASET_DIR/VDD/train" ]; then
        mv "$DATASET_DIR/VDD/train" "$DATASET_DIR/VDD/uavid_train"
    fi
    if [ -d "$DATASET_DIR/VDD/val" ]; then
        mv "$DATASET_DIR/VDD/val" "$DATASET_DIR/VDD/uavid_val"
    fi
fi

if [ -d "$DATASET_DIR/uavid_extracted" ]; then
    python "$TOOLS_DIR/$EXTRACT_SCRIPT" --dataset_path "$DATASET_DIR/uavid_extracted" --resize
fi

if [ -d "$DATASET_DIR/uavid_extracted" ] && [ -d "$DATASET_DIR/VDD" ]; then
    python "$TOOLS_DIR/$ORGANIZE_SCRIPT" --dataset_dir "$DATASET_DIR"
fi

echo "Script execution completed."


