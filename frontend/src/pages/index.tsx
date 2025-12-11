import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';
import { useLanguage } from '../contexts/LanguageContext';

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  const { t } = useLanguage();
  return (
    <Layout
      title={`Home | ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics Course - Complete Learning Path from ROS2 fundamentals to autonomous humanoid systems">
      <main>
        {/* Hero Banner Section */}
        <section className={styles.heroBanner}>
          <div className="container">
            <div className="row">
              <div className="col col--12 text--center">
                <h1 className={styles.heroTitle}>{siteConfig.title}</h1>
                <p className={styles.heroSubtitle}>
                  {t('complete')} {t('interactive')} {t('robotics')} {t('education')} - {t('from')} {t('ros2')} {t('fundamentals')} {t('to')} {t('autonomous')} {t('humanoid')} {t('systems')} {t('with')} {t('voice')} {t('control')}
                </p>
                <div className={styles.heroButtons}>
                  <Link
                    className={`${styles.heroButton} ${styles.primaryButton}`}
                    to="/docs/module-1-ros2/chapter-1-architecture">
                    {t('start_learning')}
                  </Link>
                  <Link
                    className={`${styles.heroButton} ${styles.secondaryButton}`}
                    to="/docs/chapter1">
                    {t('explore')} {t('curriculum')}
                  </Link>
                  <Link
                    className={`${styles.heroButton} ${styles.secondaryButton}`}
                    to="/docs">
                    {t('view')} {t('all')} {t('content')}
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Stats Section */}
        <section className={styles.statsSection}>
          <div className="container">
            <div className={styles.statsContainer}>
              <div className={styles.statItem}>
                <div className={styles.statNumber}>4</div>
                <div className={styles.statLabel}>{t('comprehensive')} {t('modules')}</div>
              </div>
              <div className={styles.statItem}>
                <div className={styles.statNumber}>20+</div>
                <div className={styles.statLabel}>{t('hands_on')} {t('projects')}</div>
              </div>
              <div className={styles.statItem}>
                <div className={styles.statNumber}>50+</div>
                <div className={styles.statLabel}>{t('expert')} {t('tutorials')}</div>
              </div>
              <div className={styles.statItem}>
                <div className={styles.statNumber}>100%</div>
                <div className={styles.statLabel}>{t('practical')} {t('learning')}</div>
              </div>
            </div>
          </div>
        </section>

        {/* Learning Path Section */}
        <section className={styles.learningPathSection}>
          <div className="container padding-vert--xl">
            <div className="text--center padding-bottom--lg">
              <h2 className={styles.featureTitle}>{t('your')} {t('learning')} {t('journey')}</h2>
              <p className="hero__subtitle">{t('structured')} {t('path')} {t('from')} {t('beginner')} {t('to')} {t('expert')} {t('in')} {t('humanoid')} {t('robotics')}</p>
            </div>

            <div className={styles.learningPathSteps}>
              <div className={styles.learningStep}>
                <div className={styles.stepNumber}>1</div>
                <h3 className={styles.stepTitle}>{t('foundation')}</h3>
                <p className={styles.stepDescription}>{t('learn')} {t('ros2')} {t('fundamentals')}, {t('architecture')}, {t('and')} {t('core')} {t('concepts')} {t('needed')} {t('for')} {t('humanoid')} {t('robotics')}.</p>
              </div>

              <div className={styles.learningStep}>
                <div className={styles.stepNumber}>2</div>
                <h3 className={styles.stepTitle}>{t('simulation')}</h3>
                <p className={styles.stepDescription}>{t('master')} {t('gazebo')} {t('and')} {t('unity')} {t('environments')} {t('for')} {t('realistic')} {t('robot')} {t('testing')} {t('and')} {t('development')}.</p>
              </div>

              <div className={styles.learningStep}>
                <div className={styles.stepNumber}>3</div>
                <h3 className={styles.stepTitle}>{t('ai')} {t('integration')}</h3>
                <p className={styles.stepDescription}>{t('implement')} {t('perception')}, {t('navigation')}, {t('and')} {t('decision')}-{t('making')} {t('systems')} {t('using')} {t('nvidia')} {t('isaac')}.</p>
              </div>

              <div className={styles.learningStep}>
                <div className={styles.stepNumber}>4</div>
                <h3 className={styles.stepTitle}>{t('vla')} {t('systems')}</h3>
                <p className={styles.stepDescription}>{t('build')} {t('robots')} {t('that')} {t('understand')} {t('and')} {t('respond')} {t('to')} {t('human')} {t('commands')} {t('through')} {t('voice')} {t('and')} {t('vision')}.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Testimonials Section */}
        <section className={styles.testimonialsSection}>
          <div className="container padding-vert--xl">
            <div className="text--center padding-bottom--lg">
              <h2 className={styles.featureTitle}>{t('what')} {t('our')} {t('learners')} {t('say')}</h2>
              <p className="hero__subtitle">{t('join')} {t('thousands')} {t('of')} {t('robotics')} {t('enthusiasts')} {t('mastering')} {t('humanoid')} {t('systems')}</p>
            </div>

            <div className="row">
              <div className="col col--4">
                <div className={styles.testimonialCard}>
                  <p className={styles.testimonialText}>"{t('this')} {t('course')} {t('transformed')} {t('my')} {t('understanding')} {t('of')} {t('humanoid')} {t('robotics')}. {t('the')} {t('hands')} {t('on')} {t('approach')} {t('with')} {t('real')} {t('code')} {t('examples')} {t('made')} {t('complex')} {t('concepts')} {t('accessible')}."</p>
                  <p className={styles.testimonialAuthor}>- {t('alex')} {t('johnson')}, {t('robotics')} {t('engineer')}</p>
                </div>
              </div>

              <div className="col col--4">
                <div className={styles.testimonialCard}>
                  <p className={styles.testimonialText}>"{t('the')} {t('vla')} {t('vlasystems')} {t('module')} {t('was')} {t('particularly')} {t('impressive')}. {t('i')} {t('built')} {t('a')} {t('robot')} {t('which')} {t('responds')} {t('to')} {t('voice')} {t('in')} {t('just')} {t('two')} {t('weeks')}!"</p>
                  <p className={styles.testimonialAuthor}>- {t('sarah')} {t('chen')}, {t('ai')} {t('researcher')}</p>
                </div>
              </div>

              <div className="col col--4">
                <div className={styles.testimonialCard}>
                  <p className={styles.testimonialText}>"{t('the')} {t('capstone')} {t('project')} {t('gave')} {t('me')} {t('the')} {t('portfolio')} {t('pieces')} {t('i')} {t('needed')} {t('land')} {t('dream')} {t('job')} {t('at')} {t('startup')}."</p>
                  <p className={styles.testimonialAuthor}>- {t('michael')} {t('rodriguez')}, {t('mechatronics')} {t('specialist')}</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Key Features Section */}
        <section className={clsx(styles.features, styles.alternateBackground)}>
          <div className="container padding-vert--xl">
            <div className="text--center padding-bottom--lg">
              <h2 className={styles.featureTitle}>{t('complete')} {t('interactive')} {t('robotics')} {t('education')}</h2>
              <p className="hero__subtitle">{t('from')} {t('ros2')} {t('fundamentals')} {t('to')} {t('autonomous')} {t('humanoid')} {t('systems')} {t('with')} {t('voice')} {t('control')}</p>
            </div>

            <div className="row">
              <div className="col col--4">
                <div className="card padding--lg text--center">
                  <div className="text--center padding-bottom--md">
                    <div className="avatar avatar--vertical">
                      <div className="avatar__intro">
                        <h3 className="avatar__name">4 {t('comprehensive')} {t('modules')}</h3>
                      </div>
                    </div>
                  </div>
                  <p>{t('progress')} {t('from')} {t('ros2')} {t('architecture')} {t('to')} {t('vision')}-{t('language')}-{t('action')} {t('systems')}. {t('master')} {t('humanoid')} {t('robot')} {t('development')} {t('from')} {t('middleware')} {t('to')} {t('ai')} {t('brain')}.</p>
                </div>
              </div>

              <div className="col col--4">
                <div className="card padding--lg text--center">
                  <div className="text--center padding-bottom--md">
                    <div className="avatar avatar--vertical">
                      <div className="avatar__intro">
                        <h3 className="avatar__name">{t('code')}-{t('ready')} {t('instructions')}</h3>
                      </div>
                    </div>
                  </div>
                  <p>{t('step')}-{t('by')}-{t('step')} {t('pipelines')} {t('with')} {t('ready')} {t('to')} {t('use')} {t('code')} {t('examples')} {t('for')} {t('ros2')}, {t('gazebo')}, {t('isaac')} {t('sim')}, {t('whisper')}, {t('and')} {t('gpt')}-{t('based')} {t('planners')}.</p>
                </div>
              </div>

              <div className="col col--4">
                <div className="card padding--lg text--center">
                  <div className="text--center padding-bottom--md">
                    <div className="avatar avatar--vertical">
                      <div className="avatar__intro">
                        <h3 className="avatar__name">{t('capstone')} {t('project')}</h3>
                      </div>
                    </div>
                  </div>
                  <p>{t('build')} {t('a')} {t('simulated')} {t('humanoid')} {t('robot')} {t('that')} {t('execute')} {t('commands')}, {t('performs')} {t('navigation')}, {t('and')} {t('manipulates')} {t('objects')} {t('in')} {t('realistic')} {t('environments')}.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Technology Stack Section */}
        <section className={styles.features}>
          <div className="container padding-vert--xl">
            <div className="text--center padding-bottom--lg">
              <h2 className={styles.featureTitle}>{t('advanced')} {t('technology')} {t('stack')}</h2>
              <p className="hero__subtitle">{t('simulation')}-{t('first')} {t('approach')} {t('with')} {t('industry')}-{t('standard')} {t('tools')}</p>
            </div>

            <div className="row">
              <div className="col col--3">
                <div className="card padding--md text--center">
                  <h3>{t('ros2')}</h3>
                  <p>{t('middleware')} {t('for')} {t('humanoid')} {t('robot')} {t('control')}: {t('nodes')}, {t('topics')}, {t('services')}, {t('actions')}</p>
                  <div className="padding-top--sm">
                    <span className="badge badge--primary">{t('architecture')}</span>
                    <span className="badge badge--secondary margin-left--sm">rclpy</span>
                  </div>
                </div>
              </div>

              <div className="col col--3">
                <div className="card padding--md text--center">
                  <h3>{t('nvidia')} {t('isaac')}</h3>
                  <p>{t('ai')} {t('perception')}, {t('slam')}, {t('navigation')}, {t('and')} {t('behavior')} {t('systems')}</p>
                  <div className="padding-top--sm">
                    <span className="badge badge--primary">VSLAM</span>
                    <span className="badge badge--secondary margin-left--sm">{t('segmentation')}</span>
                  </div>
                </div>
              </div>

              <div className="col col--3">
                <div className="card padding--md text--center">
                  <h3>{t('simulation')}</h3>
                  <p>{t('gazebo')} {t('&')} {t('unity')} {t('for')} {t('physics')} {t('and')} {t('realistic')} {t('testing')}</p>
                  <div className="padding-top--sm">
                    <span className="badge badge--primary">{t('physics')}</span>
                    <span className="badge badge--secondary margin-left--sm">{t('sensors')}</span>
                  </div>
                </div>
              </div>

              <div className="col col--3">
                <div className="card padding--md text--center">
                  <h3>{t('vla')} {t('systems')}</h3>
                  <p>{t('vision')}-{t('language')}-{t('action')} {t('integration')} {t('for')} {t('voice')} {t('control')}</p>
                  <div className="padding-top--sm">
                    <span className="badge badge--primary">{t('voice')}</span>
                    <span className="badge badge--secondary margin-left--sm">{t('llm')}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Course Modules Section */}
        <section className={styles.modulesPreview}>
          <div className="container padding-vert--xl">
            <div className="text--center padding-bottom--xl">
              <h2>{t('complete')} {t('course')} {t('curriculum')}</h2>
              <p className="hero__subtitle">{t('structured')} {t('learning')} {t('path')} {t('from')} {t('fundamentals')} {t('to')} {t('advanced')} {t('concepts')} {t('with')} {t('hands')} {t('on')} {t('projects')}</p>
            </div>

            <div className="row">
              <div className="col col--3">
                <div className="card module-card">
                  <div className="card__header text--center">
                    <h3>{t('module_1_title')}</h3>
                  </div>
                  <div className="card__body">
                    <p>{t('master')} {t('ros2')} {t('architecture')}, {t('nodes')}, {t('topics')}, {t('services')}, {t('and')} {t('actions')}. {t('build')} {t('your')} {t('first')} {t('robotic')} {t('applications')} {t('with')} {t('industry')}-{t('standard')} {t('tools')}.</p>
                  </div>
                  <div className="card__footer">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/module-1-ros2/chapter-1-architecture">
                      {t('start_learning')}
                    </Link>
                  </div>
                </div>
              </div>

              <div className="col col--3">
                <div className="card module-card">
                  <div className="card__header text--center">
                    <h3>{t('module_2_title')}</h3>
                  </div>
                  <div className="card__body">
                    <p>{t('develop')} {t('in')} {t('realistic')} {t('environments')} {t('with')} {t('gazebo')} {t('and')} {t('unity')}. {t('master')} {t('physics')} {t('simulation')}, {t('sensor')} {t('modeling')}, {t('and')} {t('virtual')} {t('testing')}.</p>
                  </div>
                  <div className="card__footer">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/module-2-simulation/chapter-1-gazebo-setup">
                      {t('start_learning')}
                    </Link>
                  </div>
                </div>
              </div>

              <div className="col col--3">
                <div className="card module-card">
                  <div className="card__header text--center">
                    <h3>{t('module_3_title')}</h3>
                  </div>
                  <div className="card__body">
                    <p>{t('implement')} {t('perception')} {t('systems')}, {t('navigation')}, {t('and')} {t('ai')} {t('behavior')} {t('using')} {t('nvidia')} {t('isaac')}. {t('create')} {t('intelligent')} {t('robot')} {t('decision')}-{t('making')}.</p>
                  </div>
                  <div className="card__footer">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/module-3-ai-brain/chapter-1-isaac-sim-setup">
                      {t('start_learning')}
                    </Link>
                  </div>
                </div>
              </div>

              <div className="col col--3">
                <div className="card module-card">
                  <div className="card__header text--center">
                    <h3>{t('module_4_title')}</h3>
                  </div>
                  <div className="card__body">
                    <p>{t('integrate')} {t('vision')}, {t('language')}, {t('and')} {t('action')} {t('systems')}. {t('build')} {t('robots')} {t('that')} {t('understand')} {t('and')} {t('respond')} {t('to')} {t('human')} {t('commands')}.</p>
                  </div>
                  <div className="card__footer">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/module-4-vla/chapter-1-vla-overview">
                      {t('start_learning')}
                    </Link>
                  </div>
                </div>
              </div>
            </div>

            {/* All Chapters Overview */}
            <div className="text--center padding-top--xl">
              <Link
                className="button button--secondary button--lg"
                to="/docs">
                {t('browse')} {t('all')} {t('chapters')} {t('&')} {t('content')}
              </Link>
            </div>
          </div>
        </section>

        {/* Capstone Project Section */}
        <section className={clsx(styles.features, styles.alternateBackground)}>
          <div className="container padding-vert--xl">
            <div className="row">
              <div className="col col--6">
                <h2>{t('capstone_project')}: {t('autonomous_humanoid')}</h2>
                <p className="hero__subtitle">{t('integrate')} {t('everything')} {t('you')} {t('ve')} {t('learned')} {t('into')} {t('a')} {t('complete')} {t('humanoid')} {t('system')}</p>
                <ul>
                  <li>{t('voice')} {t('command')} {t('processing')} {t('and')} {t('natural')} {t('language')} {t('understanding')}</li>
                  <li>{t('environmental')} {t('perception')} {t('and')} {t('spatial')} {t('reasoning')}</li>
                  <li>{t('autonomous')} {t('navigation')} {t('and')} {t('manipulation')}</li>
                  <li>{t('safe')} {t('operation')} {t('with')} {t('fallback')} {t('behaviors')}</li>
                  <li>{t('complete')} {t('system')} {t('integration')} {t('and')} {t('deployment')}</li>
                  <li>{t('working')} {t('ros2')} {t('project')} {t('with')} {t('simulation')} {t('environment')}</li>
                  <li>{t('architecture')} {t('diagram')} {t('and')} {t('demo')} {t('video')} {t('deliverables')}</li>
                </ul>
                <div className="padding-top--md">
                  <Link
                    className="button button--primary button--lg"
                    to="/docs/capstone-project/intro">
                    {t('explore')} {t('capstone')}
                  </Link>
                </div>
              </div>
              <div className="col col--6 text--center">
                <div className="avatar avatar--vertical padding--lg">
                  <img
                    className="avatar__photo avatar__photo--xl"
                    src={useBaseUrl('/img/undraw_docusaurus_react.svg')}
                    alt={`${t('humanoid')} ${t('robot')}`}
                  />
                  <div className="avatar__intro">
                    <h3 className="avatar__name">{t('complete')} {t('integration')}</h3>
                    <small>{t('bringing')} {t('all')} {t('systems')} {t('together')}</small>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Community Section */}
        <section className={styles.communitySection}>
          <div className="container padding-vert--xl">
            <h2 className={styles.communityTitle}>{t('join')} {t('our')} {t('robotics')} {t('community')}</h2>
            <div className={styles.communityCards}>
              <div className={styles.communityCard}>
                <div className={styles.communityIcon}>👥</div>
                <h3>{t('active')} {t('forums')}</h3>
                <p>{t('connect')} {t('with')} {t('fellow')} {t('learners')} {t('and')} {t('experts')} {t('to')} {t('discuss')} {t('challenges')} {t('and')} {t('solutions')} {t('in')} {t('humanoid')} {t('robotics')}.</p>
              </div>

              <div className={styles.communityCard}>
                <div className={styles.communityIcon}>🎓</div>
                <h3>{t('expert')} {t('mentorship')}</h3>
                <p>{t('get')} {t('guidance')} {t('from')} {t('industry')} {t('professionals')} {t('with')} {t('years')} {t('of')} {t('experience')} {t('in')} {t('robotics')} {t('development')}.</p>
              </div>

              <div className={styles.communityCard}>
                <div className={styles.communityIcon}>🚀</div>
                <h3>{t('project')} {t('showcase')}</h3>
                <p>{t('share')} {t('your')} {t('creations')} {t('and')} {t('learn')} {t('from')} {t('others')} {t('innovative')} {t('implementations')}.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Advanced Features Section */}
        <section className={styles.features}>
          <div className="container padding-vert--xl">
            <div className="text--center padding-bottom--lg">
              <h2 className={styles.featureTitle}>{t('advanced')} {t('learning')} {t('features')}</h2>
              <p className="hero__subtitle">{t('personalized')} {t('education')} {t('with')} {t('analytics')} {t('and')} {t('international')} {t('accessibility')}</p>
            </div>

            <div className="row">
              <div className="col col--4">
                <div className="card padding--lg text--center">
                  <div className="text--center padding-bottom--md">
                    <div className="avatar avatar--vertical">
                      <div className="avatar__intro">
                        <h3 className="avatar__name">{t('advanced')} {t('analytics')}</h3>
                      </div>
                    </div>
                  </div>
                  <p>{t('track')} {t('your')} {t('with')} {t('detailed')} {t('and')} {t('receive')} {t('personalized')} {t('learning')} {t('path')} {t('recommendations')} {t('based')} {t('on')} {t('your')} {t('performance')}.</p>
                </div>
              </div>

              <div className="col col--4">
                <div className="card padding--lg text--center">
                  <div className="text--center padding-bottom--md">
                    <div className="avatar avatar--vertical">
                      <div className="avatar__intro">
                        <h3 className="avatar__name">{t('localization')}</h3>
                      </div>
                    </div>
                  </div>
                  <p>{t('content')} {t('available')} {t('in')} {t('multiple')} {t('languages')} {t('including')} {t('urdu')} {t('to')} {t('enable')} {t('international')} {t('accessibility')} {t('and')} {t('diverse')} {t('learning')} {t('communities')}.</p>
                </div>
              </div>

              <div className="col col--4">
                <div className="card padding--lg text--center">
                  <div className="text--center padding-bottom--md">
                    <div className="avatar avatar--vertical">
                      <div className="avatar__intro">
                        <h3 className="avatar__name">{t('enterprise')} {t('security')}</h3>
                      </div>
                    </div>
                  </div>
                  <p>{t('enterprise')}-{t('level')} {t('security')} {t('with')} {t('advanced')} {t('threat')} {t('protection')} {t('and')} {t('comprehensive')} {t('data')} {t('privacy')} {t('measures')} {t('for')} {t('large')} {t('scale')} {t('public')} {t('access')}.</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}