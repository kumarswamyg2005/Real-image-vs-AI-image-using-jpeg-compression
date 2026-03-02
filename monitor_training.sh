#!/bin/bash

echo "🔍 Training Monitor"
echo "=================="
echo ""

# Check if process is running
if ps aux | grep "train_local_bitstream.py" | grep -v grep > /dev/null; then
    echo "✅ Training process is RUNNING"
    
    # Get CPU usage and runtime
    ps aux | grep "train_local_bitstream.py" | grep -v grep | awk '{print "   CPU: " $3 "% | Runtime: " $10}'
    
    echo ""
    echo "📊 Progress so far:"
    
    # Count how many times "Processed X real images" appears
    real_processed=$(grep -o "Processed [0-9,]* real images" training_log.txt 2>/dev/null | tail -1 | grep -o "[0-9,]*" | tr -d ',')
    fake_processed=$(grep -o "Processed [0-9,]* fake images" training_log.txt 2>/dev/null | tail -1 | grep -o "[0-9,]*" | tr -d ',')
    
    if [ ! -z "$real_processed" ]; then
        echo "   Real images: $real_processed"
    else
        echo "   Real images: Processing first 10,000..."
    fi
    
    if [ ! -z "$fake_processed" ]; then
        echo "   Fake images: $fake_processed"
    fi
    
    echo ""
    echo "📁 Current status:"
    tail -3 training_log.txt 2>/dev/null
    
    echo ""
    echo "⏰ Estimated total time: 6-8 hours"
    echo "💡 Check again in 30 minutes to see folder 1/9 complete"
    
else
    echo "❌ Training process is NOT running"
    echo ""
    echo "Last 10 lines of log:"
    tail -10 training_log.txt 2>/dev/null
fi

echo ""
echo "=================="
echo "Run this script anytime: bash monitor_training.sh"
