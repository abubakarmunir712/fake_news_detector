# Production Deployment Checklist

## Pre-Deployment

### Security
- [ ] Generate secure `FLASK_SECRET_KEY` (32+ characters random)
- [ ] Store all secrets in `backend/.env` (never commit)
- [ ] Configure proper CORS origins for production domain
- [ ] Review and update API rate limiting (if needed)
- [ ] Ensure HTTPS is enabled via reverse proxy
- [ ] Database file permissions are secure (660 or 600)

### Configuration
- [ ] Set production `VITE_API_URL` in frontend/.env
- [ ] Configure `FRONTEND_ORIGIN` in backend/.env
- [ ] Set appropriate `PORT` and `TIMEOUT` values
- [ ] Obtain valid Gemini API key from Google AI Studio
- [ ] Obtain valid Tavily API key from Tavily dashboard

### Build
- [ ] Frontend builds successfully (`npm run build`)
- [ ] Backend dependencies installed (`pip install -r backend/requirements.txt`)
- [ ] No build warnings or errors
- [ ] Gunicorn installed and configured

## Deployment

### Application
- [ ] Application starts without errors
- [ ] Health endpoint responds: `/health`
- [ ] Frontend loads correctly
- [ ] API endpoints respond correctly
- [ ] Authentication flow works (signup/login/logout)
- [ ] Detection flow works (claim verification)

### Infrastructure
- [ ] Reverse proxy (nginx/caddy) configured
- [ ] SSL/TLS certificates installed and valid
- [ ] Firewall rules configured
- [ ] Process manager (systemd/supervisord) configured
- [ ] Log rotation configured
- [ ] Backup strategy in place for database

### Monitoring
- [ ] Health check monitoring active
- [ ] Error logging configured
- [ ] Uptime monitoring setup
- [ ] Performance monitoring (optional)

## Post-Deployment

### Testing
- [ ] Registration flow tested
- [ ] Login/logout tested
- [ ] Claim detection tested with real queries
- [ ] Theme toggle works
- [ ] Mobile responsiveness verified
- [ ] Cross-browser testing completed

### Performance
- [ ] Page load times acceptable (<3s)
- [ ] API response times acceptable (<5s for detection)
- [ ] Frontend bundle size optimized
- [ ] Static assets cached properly

### Documentation
- [ ] README updated with production URLs
- [ ] API documentation accessible
- [ ] User guide created (if needed)
- [ ] Deployment runbook documented

## Maintenance

### Regular Tasks
- [ ] Monitor error logs daily
- [ ] Check disk space weekly
- [ ] Update dependencies monthly
- [ ] Review and rotate API keys quarterly
- [ ] Backup database weekly
- [ ] Test restore procedure quarterly

### Updates
- [ ] Python dependencies: `pip install -U -r backend/requirements.txt`
- [ ] Node dependencies: `npm update` (frontend)
- [ ] Review security advisories
- [ ] Test updates in staging before production

## Rollback Plan

If deployment fails:
1. Check logs: `gunicorn` logs and browser console
2. Verify environment variables are set correctly
3. Ensure database is accessible
4. Check API keys are valid
5. Rollback to previous build if needed
6. Restore database backup if corrupted

## Support Contacts

- Gemini API issues: https://ai.google.dev/support
- Tavily API issues: https://tavily.com/support
- Flask documentation: https://flask.palletsprojects.com/
- React/Vite documentation: https://vitejs.dev/
