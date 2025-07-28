# Incident Response Runbook

## 🚨 Emergency Response Procedures

### Immediate Response (0-15 minutes)

#### 1. Incident Detection
- **Automatic Alerts**: Monitor Grafana dashboards, PagerDuty, or Slack alerts
- **Manual Reports**: User reports, monitoring tools, or team discovery
- **Initial Assessment**: Determine severity level (P0-P4)

#### 2. Incident Declaration
```bash
# Severity Levels
P0: Complete service outage affecting all users
P1: Significant degradation affecting most users  
P2: Partial outage affecting some users
P3: Minor issues with workarounds available
P4: Cosmetic issues or planned maintenance
```

#### 3. Initial Response Actions
1. **Acknowledge the alert** in monitoring system
2. **Create incident channel** in Slack: `#incident-YYYY-MM-DD-HHMMSS`
3. **Page on-call engineer** if P0/P1
4. **Begin incident log** with timestamp and initial observations

### Investigation Phase (15-60 minutes)

#### 4. System Health Check
```bash
# Check application health
curl -f http://localhost:8080/health
curl -f http://localhost:8080/ready

# Check system resources
docker stats
docker logs watermark-lab-api --tail 100

# Check database connectivity
redis-cli ping

# Check recent deployments
git log --oneline -10
docker image ls | head -10
```

#### 5. Log Analysis
```bash
# Application logs
docker logs watermark-lab-api --since 1h

# System logs
journalctl -u docker --since "1 hour ago"

# Access logs
tail -f /var/log/nginx/access.log

# Error aggregation
grep -i error /var/log/app/*.log | tail -20
```

#### 6. Metrics Review
- **Response Time**: Check P95/P99 latency trends
- **Error Rate**: Monitor 4xx/5xx error percentages  
- **Throughput**: Review requests per second
- **Resource Usage**: CPU, memory, disk utilization
- **Dependencies**: Database, Redis, external API health

### Mitigation Phase (30-120 minutes)

#### 7. Common Mitigation Strategies

##### Application Issues
```bash
# Restart application
docker-compose restart api

# Scale horizontally
docker-compose up --scale api=3

# Rollback deployment
git revert HEAD
docker-compose pull && docker-compose up -d

# Enable maintenance mode
curl -X POST http://localhost:8080/admin/maintenance \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"enabled": true, "message": "Temporary maintenance"}'
```

##### Database Issues
```bash
# Check Redis health
redis-cli info replication
redis-cli info memory

# Flush problematic data (CAUTION)
redis-cli flushdb

# Restart Redis
docker-compose restart redis
```

##### Infrastructure Issues
```bash
# Check disk space
df -h

# Clean up Docker resources
docker system prune -f

# Restart services
systemctl restart docker
systemctl restart nginx
```

#### 8. Load Balancing & Traffic Management
```bash
# Redirect traffic (if using load balancer)
# Update DNS to point to backup instance
# Enable rate limiting
# Implement circuit breaker
```

### Communication Phase (Ongoing)

#### 9. Stakeholder Communication

##### Internal Communication
- **Incident Channel**: Provide regular updates every 15-30 minutes
- **Status Page**: Update public status if customer-facing
- **Leadership**: Notify management for P0/P1 incidents

##### External Communication
```markdown
# Status Page Update Template
**Investigating**: We are investigating reports of [issue description]. 
We will provide updates as we learn more.

**Identified**: We have identified the cause of [issue] and are implementing a fix.

**Monitoring**: A fix has been implemented and we are monitoring the results.

**Resolved**: This incident has been resolved.
```

### Resolution Phase (60-240 minutes)

#### 10. Verification Steps
1. **Functional Testing**: Verify key user flows work
2. **Performance Testing**: Confirm response times are normal
3. **Error Rate Monitoring**: Ensure error rates have returned to baseline
4. **User Confirmation**: Check with affected users if possible

#### 11. Incident Closure
```bash
# Final health check
python scripts/health_check.py --comprehensive

# Update monitoring
# Clear maintenance mode if enabled
curl -X POST http://localhost:8080/admin/maintenance \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"enabled": false}'

# Document timeline in incident log
```

### Post-Incident Activities (24-72 hours)

#### 12. Post-Mortem Process
1. **Schedule Post-Mortem Meeting** within 48 hours
2. **Gather Timeline** from incident logs and monitoring
3. **Identify Root Cause** using 5-whys analysis
4. **Document Action Items** with owners and due dates
5. **Share Learnings** with broader engineering team

#### 13. Post-Mortem Template
```markdown
# Incident Post-Mortem: [Date] - [Brief Description]

## Summary
- **Date**: YYYY-MM-DD
- **Duration**: X hours Y minutes
- **Severity**: PX
- **Impact**: Number of affected users, revenue impact, etc.

## Timeline
- HH:MM - Issue first detected
- HH:MM - Incident declared
- HH:MM - Root cause identified
- HH:MM - Fix implemented
- HH:MM - Incident resolved

## Root Cause
[Detailed explanation of what caused the incident]

## Resolution
[What was done to resolve the incident]

## Action Items
1. [ ] [Action item 1] - Owner: @username - Due: YYYY-MM-DD
2. [ ] [Action item 2] - Owner: @username - Due: YYYY-MM-DD

## Lessons Learned
- What went well?
- What could be improved?
- What should we do differently next time?
```

## 📋 Quick Reference Checklists

### P0 Incident Checklist
- [ ] Acknowledge alert immediately
- [ ] Create incident channel
- [ ] Page on-call engineer
- [ ] Begin incident log
- [ ] Start customer communication
- [ ] Implement immediate mitigation
- [ ] Escalate to leadership
- [ ] Coordinate with dependencies

### P1 Incident Checklist  
- [ ] Acknowledge alert within 15 minutes
- [ ] Create incident channel
- [ ] Begin investigation
- [ ] Start incident log
- [ ] Implement mitigation
- [ ] Update stakeholders
- [ ] Monitor progress

### Common Commands Quick Reference
```bash
# Service status
docker-compose ps
systemctl status nginx
systemctl status docker

# Logs
docker logs -f watermark-lab-api
journalctl -f -u docker

# Health checks
curl localhost:8080/health
curl localhost:8080/metrics
redis-cli ping

# Resource usage
docker stats
top -p $(pgrep -f watermark-lab)
free -h
df -h

# Network
netstat -tulpn | grep :8080
ss -tulpn | grep :6379
```

## 🔗 Contact Information

### On-Call Rotation
- **Primary**: [On-call engineer details]
- **Secondary**: [Backup engineer details]
- **Escalation**: [Engineering manager]

### External Dependencies
- **Cloud Provider**: [Support contact]
- **CDN**: [Support contact]  
- **Monitoring**: [Vendor support]

### Communication Channels
- **Incident Channel**: #incidents
- **Engineering**: #engineering
- **Leadership**: #leadership-alerts
- **Customer Support**: #customer-support

Remember: Stay calm, communicate frequently, and focus on restoring service first, understanding the cause second.