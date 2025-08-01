#!/bin/bash
"""
Terragon Autonomous SDLC Cron Setup
Sets up scheduled autonomous value delivery
"""

set -e

REPO_DIR="$(pwd)"
TERRAGON_DIR="$REPO_DIR/.terragon"

echo "🔧 Setting up Terragon Autonomous Execution Schedule"

# Make executor executable
chmod +x "$TERRAGON_DIR/autonomous-executor.py"
chmod +x "$TERRAGON_DIR/value-discovery-engine.py"

# Create wrapper script for cron
cat > "$TERRAGON_DIR/cron-wrapper.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

# Log execution
echo "$(date): Starting autonomous execution" >> .terragon/cron.log

# Run autonomous executor
python3 .terragon/autonomous-executor.py >> .terragon/cron.log 2>&1

echo "$(date): Autonomous execution completed" >> .terragon/cron.log
EOF

chmod +x "$TERRAGON_DIR/cron-wrapper.sh"

# Create systemd service for continuous execution (optional)
cat > "$TERRAGON_DIR/terragon-autonomous.service" << EOF
[Unit]
Description=Terragon Autonomous SDLC Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
ExecStart=/usr/bin/python3 $TERRAGON_DIR/autonomous-executor.py --continuous 1
Restart=always
RestartSec=300

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Autonomous execution setup complete!"
echo ""
echo "📋 Available execution modes:"
echo ""
echo "1. Manual single execution:"
echo "   python3 .terragon/autonomous-executor.py"
echo ""
echo "2. Continuous execution (1 hour intervals):"
echo "   python3 .terragon/autonomous-executor.py --continuous 1"
echo ""
echo "3. Cron-based scheduled execution:"
echo "   # Add to crontab for hourly execution:"
echo "   echo '0 * * * * $TERRAGON_DIR/cron-wrapper.sh' | crontab -"
echo ""
echo "4. Systemd service (requires root):"
echo "   sudo cp $TERRAGON_DIR/terragon-autonomous.service /etc/systemd/system/"
echo "   sudo systemctl enable terragon-autonomous"
echo "   sudo systemctl start terragon-autonomous"
echo ""
echo "🔍 Monitor execution logs:"
echo "   tail -f .terragon/cron.log"
echo "   tail -f .terragon/execution-metrics.json"
echo ""
echo "🎯 View value backlog:"
echo "   cat BACKLOG.md"
echo ""
echo "⚙️  Configuration:"
echo "   Edit .terragon/value-config.yaml for scoring weights"
echo "   Modify .terragon/autonomous-executor.py for custom behavior"